#include <vkgs/engine/engine.h>

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <mutex>
#include <vector>
#include <map>
#include <fstream>

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"

#include <glm/gtc/type_ptr.hpp>

#include "vkgs/viewer/viewer.h"

#include <vkgs/scene/camera.h>
#include <vkgs/util/clock.h>

#include "generated/color_vert.h"
#include "generated/color_frag.h"

namespace vkgs {
namespace {

struct Resolution {
  int width;
  int height;
  const char* tag;
};

std::vector<Resolution> presetResolutions = {
    // clang-format off
    {640, 480, "640 x 480 (480p)"},
    {800, 600, "800 x 600"},
    {1280, 720, "1280 x 720 (720p, HD)"},
    {1600, 900, "1600 x 900"},
    {1920, 1080, "1920 x 1080 (1080p, FHD)"},
    {2560, 1440, "2560 x 1440 (1440p, QHD)"},
    // clang-format on
};

const std::string pipelineCacheFilename = "pipeline_cache.bin";

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                    void* pUserData) {
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl << std::endl;

  return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

struct UniformCamera {
  glm::mat4 projection;
  glm::mat4 view;
  glm::vec3 camera_position;
  alignas(16) glm::uvec2 screen_size;  // (width, height)
};

}  // namespace

class Engine::Impl {
 public:
  Impl() {
    // instance
    VkApplicationInfo applicationInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    applicationInfo.pApplicationName = "vkgs";
    applicationInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 0, 0);
    applicationInfo.pEngineName = "vkgs_vulkan";
    applicationInfo.engineVersion = VK_MAKE_API_VERSION(0, 0, 0, 0);
    applicationInfo.apiVersion = VK_API_VERSION_1_3;

    VkDebugUtilsMessengerCreateInfoEXT messengerInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messengerInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messengerInfo.pfnUserCallback = DebugCallback;

    std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};

    uint32_t count;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char*> instanceExtensions(glfwExtensions, glfwExtensions + count);
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    VkInstanceCreateInfo instanceInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pNext = &messengerInfo;
    instanceInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    instanceInfo.pApplicationInfo = &applicationInfo;
    instanceInfo.enabledLayerCount = layers.size();
    instanceInfo.ppEnabledLayerNames = layers.data();
    instanceInfo.enabledExtensionCount = instanceExtensions.size();
    instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();
    vkCreateInstance(&instanceInfo, NULL, &instance_);

    CreateDebugUtilsMessengerEXT(instance_, &messengerInfo, NULL, &messenger_);

    // physical device
    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &physicalDeviceCount, NULL);
    if (physicalDeviceCount == 0) throw std::runtime_error("No GPU found");
    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(instance_, &physicalDeviceCount, physicalDevices.data());
    physicalDevice_ = physicalDevices[0];

    // physical device properties
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice_, &physicalDeviceProperties);
    deviceName_ = physicalDeviceProperties.deviceName;
    std::cout << deviceName_ << std::endl;

    // find graphics queue
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, NULL);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, queueFamilies.data());

    // TODO: choose proper queues.
    constexpr VkQueueFlags graphicsQueueFlags = VK_QUEUE_GRAPHICS_BIT;
    constexpr VkQueueFlags transferQueueFlags = VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT;
    constexpr VkQueueFlags computeQueueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    for (int i = 0; i < queueFamilies.size(); ++i) {
      const auto& queueFamily = queueFamilies[i];

      bool isGraphicsQueueType = (queueFamily.queueFlags & graphicsQueueFlags) == graphicsQueueFlags;
      bool presentationSupport = glfwGetPhysicalDevicePresentationSupport(instance_, physicalDevice_, i);

      bool isTransferQueueType = queueFamily.queueFlags == transferQueueFlags;

      bool isComputeQueueType = (queueFamily.queueFlags & computeQueueFlags) == computeQueueFlags;

      if (isGraphicsQueueType && presentationSupport) graphicsQueueFamily_ = i;
      if (!isGraphicsQueueType && isTransferQueueType) transferQueueFamily_ = i;
      if (!isGraphicsQueueType && isComputeQueueType) computeQueueFamily_ = i;
    }

    std::cout << "transfer queue family: " << transferQueueFamily_ << std::endl;
    std::cout << "compute  queue family: " << computeQueueFamily_ << std::endl;
    std::cout << "graphics queue family: " << graphicsQueueFamily_ << std::endl;

    VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};

    VkPhysicalDevice16BitStorageFeatures k16bitStorageFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    k16bitStorageFeatures.pNext = &bufferDeviceAddressFeatures;

    VkPhysicalDeviceSynchronization2Features synchronization2Features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    synchronization2Features.pNext = &k16bitStorageFeatures;

    VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &synchronization2Features;
    vkGetPhysicalDeviceFeatures2(physicalDevice_, &features);

    // queues
    std::vector<float> queuePriorities = {0.25f, 0.5f, 1.f};
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    queueInfos.resize(3);
    queueInfos[0] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfos[0].queueFamilyIndex = computeQueueFamily_;
    queueInfos[0].queueCount = 1;
    queueInfos[0].pQueuePriorities = &queuePriorities[0];

    queueInfos[1] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfos[1].queueFamilyIndex = transferQueueFamily_;
    queueInfos[1].queueCount = 1;
    queueInfos[1].pQueuePriorities = &queuePriorities[1];

    queueInfos[2] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfos[2].queueFamilyIndex = graphicsQueueFamily_;
    queueInfos[2].queueCount = 1;
    queueInfos[2].pQueuePriorities = &queuePriorities[1];

    // device
    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#ifdef __APPLE__
        "VK_KHR_portability_subset",
#endif
    };

    VkDeviceCreateInfo deviceInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.pNext = &features;
    deviceInfo.queueCreateInfoCount = queueInfos.size();
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    deviceInfo.enabledExtensionCount = deviceExtensions.size();
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
    vkCreateDevice(physicalDevice_, &deviceInfo, NULL, &device_);

    vkGetDeviceQueue(device_, transferQueueFamily_, 0, &transferQueue_);
    vkGetDeviceQueue(device_, computeQueueFamily_, 0, &computeQueue_);
    vkGetDeviceQueue(device_, graphicsQueueFamily_, 0, &graphicsQueue_);

    // allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocatorInfo.physicalDevice = physicalDevice_;
    allocatorInfo.device = device_;
    allocatorInfo.instance = instance_;
    allocatorInfo.vulkanApiVersion = applicationInfo.apiVersion;
    vmaCreateAllocator(&allocatorInfo, &allocator_);

    // command pool
    VkCommandPoolCreateInfo commandPoolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolInfo.queueFamilyIndex = transferQueueFamily_;
    vkCreateCommandPool(device_, &commandPoolInfo, NULL, &transferCommandPool_);
    commandPoolInfo.queueFamilyIndex = computeQueueFamily_;
    vkCreateCommandPool(device_, &commandPoolInfo, NULL, &computeCommandPool_);
    commandPoolInfo.queueFamilyIndex = graphicsQueueFamily_;
    vkCreateCommandPool(device_, &commandPoolInfo, NULL, &graphicsCommandPool_);

    // descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2048},       {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 64},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048},       {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 64},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
    };
    VkDescriptorPoolCreateInfo descriptorPoolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptorPoolInfo.maxSets = 2048;
    descriptorPoolInfo.poolSizeCount = poolSizes.size();
    descriptorPoolInfo.pPoolSizes = poolSizes.data();
    vkCreateDescriptorPool(device_, &descriptorPoolInfo, NULL, &descriptorPool_);

    // pipeline cache
    std::vector<char> pipelineCacheData;
    {
      std::ifstream in(pipelineCacheFilename, std::ios::binary);
      if (in.is_open()) pipelineCacheData = std::vector<char>(std::istreambuf_iterator<char>(in), {});
    }
    VkPipelineCacheCreateInfo pipelineCacheInfo = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    pipelineCacheInfo.initialDataSize = pipelineCacheData.size();
    pipelineCacheInfo.pInitialData = pipelineCacheData.data();
    vkCreatePipelineCache(device_, &pipelineCacheInfo, NULL, &pipelineCache_);

    // synchronizations
    VkSemaphoreCreateInfo semaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    imageAcquiredSemaphores_.resize(3);
    for (int i = 0; i < 3; ++i) {
      vkCreateSemaphore(device_, &semaphoreInfo, NULL, &imageAcquiredSemaphores_[i]);
    }

    renderFinishedSemaphores_.resize(2);
    renderFinishedFences_.resize(2);
    for (int i = 0; i < 2; ++i) {
      vkCreateSemaphore(device_, &semaphoreInfo, NULL, &renderFinishedSemaphores_[i]);
      vkCreateFence(device_, &fenceInfo, NULL, &renderFinishedFences_[i]);
    }

    // command buffers
    VkCommandBufferAllocateInfo commandBufferInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandBufferInfo.commandPool = graphicsCommandPool_;
    commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferInfo.commandBufferCount = 3;
    graphicsCommandBuffers_.resize(commandBufferInfo.commandBufferCount);
    vkAllocateCommandBuffers(device_, &commandBufferInfo, graphicsCommandBuffers_.data());

    // layouts
    VkDescriptorSetLayoutBinding binding = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT};
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptorSetLayoutInfo.bindingCount = 1;
    descriptorSetLayoutInfo.pBindings = &binding;
    vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutInfo, NULL, &cameraSetLayout_);

    VkPushConstantRange pushConstantRange = {VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4)};
    VkPipelineLayoutCreateInfo graphicsPipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    graphicsPipelineLayoutInfo.setLayoutCount = 1;
    graphicsPipelineLayoutInfo.pSetLayouts = &cameraSetLayout_;
    graphicsPipelineLayoutInfo.pushConstantRangeCount = 1;
    graphicsPipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    vkCreatePipelineLayout(device_, &graphicsPipelineLayoutInfo, NULL, &graphicsPipelineLayout_);

    // descriptors and buffers
    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = sizeof(UniformCamera);
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VmaAllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    cameraBuffers_.resize(2);
    VmaAllocationInfo allocationInfo;
    vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &cameraBuffers_[0].buffer,
                    &cameraBuffers_[0].allocation, &allocationInfo);
    cameraBuffers_[0].ptr = allocationInfo.pMappedData;
    vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &cameraBuffers_[1].buffer,
                    &cameraBuffers_[1].allocation, &allocationInfo);
    cameraBuffers_[1].ptr = allocationInfo.pMappedData;

    std::vector<VkDescriptorSetLayout> cameraSetLayouts = {
        cameraSetLayout_,
        cameraSetLayout_,
    };
    VkDescriptorSetAllocateInfo descriptorSetInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    descriptorSetInfo.descriptorPool = descriptorPool_;
    descriptorSetInfo.descriptorSetCount = cameraSetLayouts.size();
    descriptorSetInfo.pSetLayouts = cameraSetLayouts.data();
    cameraSets_.resize(2);
    vkAllocateDescriptorSets(device_, &descriptorSetInfo, cameraSets_.data());

    std::vector<VkDescriptorBufferInfo> bufferInfos(2);
    bufferInfos[0] = {cameraBuffers_[0].buffer, 0, sizeof(UniformCamera)};
    bufferInfos[1] = {cameraBuffers_[1].buffer, 0, sizeof(UniformCamera)};

    std::vector<VkWriteDescriptorSet> writes(2);
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[0].dstSet = cameraSets_[0];
    writes[0].dstBinding = 0;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &bufferInfos[0];

    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[1].dstSet = cameraSets_[1];
    writes[1].dstBinding = 0;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo = &bufferInfos[0];

    vkUpdateDescriptorSets(device_, writes.size(), writes.data(), 0, NULL);

    // transfer
    {
      VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bufferInfo.size = 2 * 1024 * 1024;  // 2MB
      bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      VmaAllocationCreateInfo allocationCreateInfo = {};
      allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
      allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
      VmaAllocationInfo allocationInfo;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &staging_.buffer, &staging_.allocation,
                      &allocationInfo);
      staging_.ptr = allocationInfo.pMappedData;

      VkCommandBufferAllocateInfo commandBufferInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      commandBufferInfo.commandPool = transferCommandPool_;
      commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      commandBufferInfo.commandBufferCount = 1;
      vkAllocateCommandBuffers(device_, &commandBufferInfo, &transferCommandBuffer_);

      VkSemaphoreCreateInfo semaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      vkCreateSemaphore(device_, &semaphoreInfo, NULL, &transferSemaphore_);
    }

    // objects
    {
      std::vector<float> vertexBuffer = {
          0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f,  //
          1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f,  //
          0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f,  //
      };
      std::vector<uint32_t> indexBuffer = {0, 1, 2, 0, 2, 1};

      VkDeviceSize vertexBufferSize = vertexBuffer.size() * sizeof(float);
      VkDeviceSize indexBufferSize = indexBuffer.size() * sizeof(uint32_t);

      VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bufferInfo.size = vertexBufferSize;
      bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
      VmaAllocationCreateInfo allocationCreateInfo = {};
      allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &triangle_.vertexBuffer.buffer,
                      &triangle_.vertexBuffer.allocation, NULL);

      bufferInfo.size = indexBufferSize;
      bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &triangle_.indexBuffer.buffer,
                      &triangle_.indexBuffer.allocation, NULL);

      triangle_.indexCount = indexBuffer.size();
      triangle_.model = glm::mat4(1.f);

      std::memcpy(staging_.ptr, vertexBuffer.data(), vertexBufferSize);
      std::memcpy(static_cast<uint8_t*>(staging_.ptr) + vertexBufferSize, indexBuffer.data(), indexBufferSize);

      VkCommandBufferBeginInfo commandBufferBeginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(transferCommandBuffer_, &commandBufferBeginInfo);

      VkBufferCopy region = {0, 0, vertexBufferSize};
      vkCmdCopyBuffer(transferCommandBuffer_, staging_.buffer, triangle_.vertexBuffer.buffer, 1, &region);

      region = {vertexBufferSize, 0, indexBufferSize};
      vkCmdCopyBuffer(transferCommandBuffer_, staging_.buffer, triangle_.indexBuffer.buffer, 1, &region);

      std::vector<VkBufferMemoryBarrier> barriers(2);
      barriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      barriers[0].buffer = triangle_.vertexBuffer.buffer;
      barriers[0].offset = 0;
      barriers[0].size = vertexBufferSize;
      barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barriers[0].srcQueueFamilyIndex = transferQueueFamily_;

      barriers[1] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      barriers[1].buffer = triangle_.indexBuffer.buffer;
      barriers[1].offset = 0;
      barriers[1].size = indexBufferSize;
      barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barriers[1].srcQueueFamilyIndex = transferQueueFamily_;
      vkCmdPipelineBarrier(transferCommandBuffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, NULL, barriers.size(),
                           barriers.data(), 0, NULL);

      vkEndCommandBuffer(transferCommandBuffer_);

      VkCommandBufferSubmitInfo commandBufferInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
      commandBufferInfo.commandBuffer = transferCommandBuffer_;
      VkSemaphoreSubmitInfo signalSemaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      signalSemaphoreInfo.semaphore = transferSemaphore_;
      signalSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
      VkSubmitInfo2 submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submitInfo.commandBufferInfoCount = 1;
      submitInfo.pCommandBufferInfos = &commandBufferInfo;
      submitInfo.signalSemaphoreInfoCount = 1;
      submitInfo.pSignalSemaphoreInfos = &signalSemaphoreInfo;
      vkQueueSubmit2(transferQueue_, 1, &submitInfo, NULL);

      // transfer ownership
      vkBeginCommandBuffer(graphicsCommandBuffers_[0], &commandBufferBeginInfo);

      barriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      barriers[0].buffer = triangle_.vertexBuffer.buffer;
      barriers[0].offset = 0;
      barriers[0].size = vertexBufferSize;
      barriers[0].srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
      barriers[0].srcQueueFamilyIndex = transferQueueFamily_;

      barriers[1] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      barriers[1].buffer = triangle_.indexBuffer.buffer;
      barriers[1].offset = 0;
      barriers[1].size = indexBufferSize;
      barriers[1].dstAccessMask = VK_ACCESS_INDEX_READ_BIT;
      barriers[1].dstQueueFamilyIndex = transferQueueFamily_;
      vkCmdPipelineBarrier(graphicsCommandBuffers_[0], VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, NULL, barriers.size(),
                           barriers.data(), 0, NULL);

      vkEndCommandBuffer(graphicsCommandBuffers_[0]);

      VkSemaphoreSubmitInfo waitSemaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      waitSemaphoreInfo.semaphore = transferSemaphore_;
      waitSemaphoreInfo.stageMask =
          VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
      commandBufferInfo.commandBuffer = graphicsCommandBuffers_[0];
      submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submitInfo.waitSemaphoreInfoCount = 1;
      submitInfo.pWaitSemaphoreInfos = &waitSemaphoreInfo;
      submitInfo.commandBufferInfoCount = 1;
      submitInfo.pCommandBufferInfos = &commandBufferInfo;
      vkQueueSubmit2(graphicsQueue_, 1, &submitInfo, NULL);
    }
  }

  ~Impl() {
    vkDeviceWaitIdle(device_);

    size_t size;
    vkGetPipelineCacheData(device_, pipelineCache_, &size, NULL);
    if (size > 0) {
      std::vector<char> data(size);
      vkGetPipelineCacheData(device_, pipelineCache_, &size, data.data());

      std::ofstream out(pipelineCacheFilename, std::ios::binary);
      if (out.is_open()) out.write(data.data(), data.size());
    }
    vkDestroyPipelineCache(device_, pipelineCache_, NULL);

    vkDestroyDescriptorSetLayout(device_, cameraSetLayout_, NULL);
    vkDestroyPipelineLayout(device_, graphicsPipelineLayout_, NULL);
    if (colorTrianglePipeline_) vkDestroyPipeline(device_, colorTrianglePipeline_, NULL);

    for (auto semaphore : imageAcquiredSemaphores_) vkDestroySemaphore(device_, semaphore, NULL);
    for (auto semaphore : renderFinishedSemaphores_) vkDestroySemaphore(device_, semaphore, NULL);
    for (auto fence : renderFinishedFences_) vkDestroyFence(device_, fence, NULL);

    for (auto framebuffer : framebuffers_) vkDestroyFramebuffer(device_, framebuffer, NULL);

    vkDestroyRenderPass(device_, renderPass_, NULL);

    vkDestroyCommandPool(device_, transferCommandPool_, NULL);
    vkDestroyCommandPool(device_, computeCommandPool_, NULL);
    vkDestroyCommandPool(device_, graphicsCommandPool_, NULL);
    vkDestroyDescriptorPool(device_, descriptorPool_, NULL);

    vmaDestroyBuffer(allocator_, triangle_.vertexBuffer.buffer, triangle_.vertexBuffer.allocation);
    vmaDestroyBuffer(allocator_, triangle_.indexBuffer.buffer, triangle_.indexBuffer.allocation);

    vmaDestroyBuffer(allocator_, staging_.buffer, staging_.allocation);
    vkDestroySemaphore(device_, transferSemaphore_, NULL);

    for (const auto& buffer : cameraBuffers_) vmaDestroyBuffer(allocator_, buffer.buffer, buffer.allocation);
    vmaDestroyAllocator(allocator_);

    vkDestroyDevice(device_, NULL);
    DestroyDebugUtilsMessengerEXT(instance_, messenger_, NULL);
    vkDestroyInstance(instance_, NULL);
  }

  void Run() {
    viewer_.PrepareWindow(instance_);

    RecreateSwapchain();
    RecreateRenderPass();
    RecreateFramebuffer();
    RecreateUi();
    shouldRecreateSwapchain_ = false;
    shouldRecreateRenderPass_ = false;
    shouldRecreateFramebuffer_ = false;
    shouldRecreateUi_ = false;

    while (!viewer_.ShouldClose()) {
      viewer_.PollEvents();

      // draw ui
      viewer_.BeginUi();
      HandleEvents();
      DrawUi();
      viewer_.EndUi();

      DoComputeJobs();
      DoGraphicsJobs();
    }

    vkDeviceWaitIdle(device_);

    vkDestroySwapchainKHR(device_, swapchain_, NULL);
    for (auto imageView : swapchainImageViews_) vkDestroyImageView(device_, imageView, NULL);
    viewer_.DestroyWindow();
  }

 private:
  void DoComputeJobs() {
    // TODO
  }

  void DoGraphicsJobs() {
    // skip if window is not visible
    auto [width, height] = viewer_.windowSize();
    if (width == 0 || height == 0) {
      return;
    }

    if (shouldRecreateSwapchain_) {
      vkWaitForFences(device_, renderFinishedFences_.size(), renderFinishedFences_.data(), VK_TRUE, UINT64_MAX);
      RecreateSwapchain();
      shouldRecreateSwapchain_ = false;
    }

    VkSemaphore imageAcquiredSemaphore = imageAcquiredSemaphores_[imageAcquireIndex_];
    uint32_t imageIndex;
    VkResult imageAcquireResult =
        vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imageAcquiredSemaphore, NULL, &imageIndex);
    if (imageAcquireResult == VK_SUBOPTIMAL_KHR || imageAcquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
      shouldRecreateSwapchain_ = true;
    }
    if (imageAcquireResult == VK_SUCCESS || imageAcquireResult == VK_SUBOPTIMAL_KHR) {
      if (shouldRecreateRenderPass_) {
        vkWaitForFences(device_, renderFinishedFences_.size(), renderFinishedFences_.data(), VK_TRUE, UINT64_MAX);
        RecreateRenderPass();
        shouldRecreateRenderPass_ = false;
      }

      if (shouldRecreateFramebuffer_) {
        vkWaitForFences(device_, renderFinishedFences_.size(), renderFinishedFences_.data(), VK_TRUE, UINT64_MAX);
        RecreateFramebuffer();
        shouldRecreateFramebuffer_ = false;
      }

      if (shouldRecreateUi_) {
        vkWaitForFences(device_, renderFinishedFences_.size(), renderFinishedFences_.data(), VK_TRUE, UINT64_MAX);
        RecreateUi();
        shouldRecreateUi_ = false;
      }

      if (shouldRecreatePipelines_) {
        vkWaitForFences(device_, renderFinishedFences_.size(), renderFinishedFences_.data(), VK_TRUE, UINT64_MAX);
        RecreatePipelines();
        shouldRecreatePipelines_ = false;
      }

      VkCommandBuffer cb = graphicsCommandBuffers_[renderIndex_];
      VkSemaphore renderFinishedSemaphore = renderFinishedSemaphores_[renderIndex_];
      VkFence renderFinishedFence = renderFinishedFences_[renderIndex_];

      // wait for render finishes
      vkWaitForFences(device_, 1, &renderFinishedFence, VK_TRUE, UINT64_MAX);

      // record draw commands
      VkCommandBufferBeginInfo commandBufferBeginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(cb, &commandBufferBeginInfo);

      std::vector<VkClearValue> clearValues(2);
      clearValues[0].color.float32[0] = 0.f;
      clearValues[0].color.float32[1] = 0.f;
      clearValues[0].color.float32[2] = 0.f;
      clearValues[0].color.float32[3] = 1.f;
      clearValues[1].depthStencil.depth = 1.f;
      VkRenderPassBeginInfo renderPassBeginInfo = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      renderPassBeginInfo.renderPass = renderPass_;
      renderPassBeginInfo.framebuffer = framebuffers_[imageIndex];
      renderPassBeginInfo.renderArea.offset = {0, 0};
      renderPassBeginInfo.renderArea.extent = {swapchainWidth_, swapchainHeight_};
      renderPassBeginInfo.clearValueCount = clearValues.size();
      renderPassBeginInfo.pClearValues = clearValues.data();
      vkCmdBeginRenderPass(cb, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

      VkViewport viewport;
      viewport.x = 0;
      viewport.y = 0;
      viewport.width = width;
      viewport.height = height;
      viewport.minDepth = 0.f;
      viewport.maxDepth = 1.f;
      vkCmdSetViewport(cb, 0, 1, &viewport);

      VkRect2D scissor;
      scissor.offset = {0, 0};
      scissor.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
      vkCmdSetScissor(cb, 0, 1, &scissor);

      // update uniform buffer
      UniformCamera camera;
      camera.projection = camera_.ProjectionMatrix();
      camera.view = camera_.ViewMatrix();
      camera.camera_position = camera_.Eye();
      camera.screen_size = {width, height};
      std::memcpy(cameraBuffers_[renderIndex_].ptr, &camera, sizeof(UniformCamera));

      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout_, 0, 1,
                              &cameraSets_[renderIndex_], 0, NULL);
      vkCmdPushConstants(cb, graphicsPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4),
                         glm::value_ptr(triangle_.model));

      // color
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, colorTrianglePipeline_);
      VkDeviceSize vertexBufferOffset = 0;
      vkCmdBindVertexBuffers(cb, 0, 1, &triangle_.vertexBuffer.buffer, &vertexBufferOffset);
      vkCmdBindIndexBuffer(cb, triangle_.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(cb, triangle_.indexCount, 1, 0, 0, 0);

      viewer_.DrawUi(cb);

      vkCmdEndRenderPass(cb);
      vkEndCommandBuffer(cb);

      VkSemaphoreSubmitInfo waitSemaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      waitSemaphoreInfo.semaphore = imageAcquiredSemaphore;
      waitSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      VkCommandBufferSubmitInfo commandBufferInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
      commandBufferInfo.commandBuffer = cb;
      VkSemaphoreSubmitInfo signalSemaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      signalSemaphoreInfo.semaphore = renderFinishedSemaphore;
      signalSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      VkSubmitInfo2 submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submitInfo.waitSemaphoreInfoCount = 1;
      submitInfo.pWaitSemaphoreInfos = &waitSemaphoreInfo;
      submitInfo.commandBufferInfoCount = 1;
      submitInfo.pCommandBufferInfos = &commandBufferInfo;
      submitInfo.signalSemaphoreInfoCount = 1;
      submitInfo.pSignalSemaphoreInfos = &signalSemaphoreInfo;
      vkResetFences(device_, 1, &renderFinishedFence);
      vkQueueSubmit2(graphicsQueue_, 1, &submitInfo, renderFinishedFence);

      VkPresentInfoKHR presentInfo = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
      presentInfo.waitSemaphoreCount = 1;
      presentInfo.pWaitSemaphores = &renderFinishedSemaphore;
      presentInfo.swapchainCount = 1;
      presentInfo.pSwapchains = &swapchain_;
      presentInfo.pImageIndices = &imageIndex;
      VkResult presentResult = vkQueuePresentKHR(graphicsQueue_, &presentInfo);

      if (presentResult == VK_SUBOPTIMAL_KHR || presentResult == VK_ERROR_OUT_OF_DATE_KHR) {
        shouldRecreateSwapchain_ = true;
      }

      renderIndex_ = (renderIndex_ + 1) % 2;
      imageAcquireIndex_ = (imageAcquireIndex_ + 1) % 3;
    }
  }

  void HandleEvents() {
    const auto& io = ImGui::GetIO();

    if (!io.WantCaptureMouse) {
      bool left = io.MouseDown[ImGuiMouseButton_Left];
      bool right = io.MouseDown[ImGuiMouseButton_Right];
      float dx = io.MouseDelta.x;
      float dy = io.MouseDelta.y;

      if (left && !right) {
        camera_.Rotate(dx, dy);
      } else if (!left && right) {
        camera_.Translate(dx, dy);
      } else if (left && right) {
        camera_.Zoom(dy);
      }

      if (io.MouseWheel != 0.f) {
        if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl)) {
          camera_.DollyZoom(io.MouseWheel);
        } else {
          camera_.Zoom(io.MouseWheel * 10.f);
        }
      }
    }

    if (!io.WantCaptureKeyboard) {
      constexpr float speed = 1000.f;
      float dt = io.DeltaTime;
      if (ImGui::IsKeyDown(ImGuiKey_W)) {
        camera_.Translate(0.f, 0.f, speed * dt);
      }
      if (ImGui::IsKeyDown(ImGuiKey_S)) {
        camera_.Translate(0.f, 0.f, -speed * dt);
      }
      if (ImGui::IsKeyDown(ImGuiKey_A)) {
        camera_.Translate(speed * dt, 0.f);
      }
      if (ImGui::IsKeyDown(ImGuiKey_D)) {
        camera_.Translate(-speed * dt, 0.f);
      }
      if (ImGui::IsKeyDown(ImGuiKey_Space)) {
        camera_.Translate(0.f, speed * dt);
      }

      if (ImGui::IsKeyDown(ImGuiKey::ImGuiMod_Alt) && ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
        ToggleDisplayMode();
      }
    }
  }

  void DrawUi() {
    auto& io = ImGui::GetIO();
    if (ImGui::Begin("vkgs")) {
      ImGui::Text("Device: %s", deviceName_.c_str());
      ImGui::Text("FPS   : %.2f", io.Framerate);

      // display mode
      std::vector<const char*> displayModeLabels = {
          "Windowed",
          "Windowed Fullscreen",
      };
      const char* currentDisplayMode = displayModeLabels[static_cast<int>(viewer_.displayMode())];
      if (ImGui::BeginCombo("Display mode", currentDisplayMode)) {
        if (ImGui::Selectable("Windowed")) viewer_.SetWindowed();
        if (ImGui::Selectable("Windowed Fullscreen")) viewer_.SetWindowedFullscreen();
        ImGui::EndCombo();
      }

      // resolution
      auto [width, height] = viewer_.windowSize();
      std::string currentResolution = std::to_string(width) + " x " + std::to_string(height);
      ImGui::BeginDisabled(viewer_.displayMode() == viewer::DisplayMode::WindowedFullscreen);
      if (ImGui::BeginCombo("Resolution", currentResolution.c_str())) {
        for (int i = 0; i < presetResolutions.size(); ++i) {
          const auto& resolution = presetResolutions[i];
          if (ImGui::Selectable(resolution.tag)) {
            viewer_.SetWindowSize(resolution.width, resolution.height);
          }
        }
        ImGui::EndCombo();
      }
      ImGui::EndDisabled();

      // vsync
      int presentMode = swapchainPresentMode_;
      ImGui::Text("VSync");
      ImGui::SameLine();
      ImGui::RadioButton("On", &presentMode, VK_PRESENT_MODE_FIFO_KHR);
      ImGui::SameLine();
      ImGui::RadioButton("Off", &presentMode, VK_PRESENT_MODE_MAILBOX_KHR);
      if (swapchainPresentMode_ != static_cast<VkPresentModeKHR>(presentMode)) shouldRecreateSwapchain_ = true;
      swapchainPresentMode_ = static_cast<VkPresentModeKHR>(presentMode);
    }
    ImGui::End();
  }

  void RecreateSwapchain() {
    VkSurfaceKHR surface = viewer_.surface();

    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface, &surfaceCapabilities);

    VkSwapchainKHR oldSwapchain = swapchain_;
    VkSwapchainCreateInfoKHR swapchainInfo = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    swapchainInfo.surface = viewer_.surface();
    swapchainInfo.minImageCount = 3;
    swapchainInfo.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    swapchainInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swapchainInfo.imageExtent = surfaceCapabilities.currentExtent;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchainInfo.preTransform = surfaceCapabilities.currentTransform;
    swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainInfo.presentMode = swapchainPresentMode_;
    swapchainInfo.clipped = VK_TRUE;
    swapchainInfo.oldSwapchain = oldSwapchain;
    vkCreateSwapchainKHR(device_, &swapchainInfo, NULL, &swapchain_);
    if (oldSwapchain) vkDestroySwapchainKHR(device_, oldSwapchain, NULL);

    swapchainWidth_ = swapchainInfo.imageExtent.width;
    swapchainHeight_ = swapchainInfo.imageExtent.height;

    uint32_t imageCount;
    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, NULL);
    std::vector<VkImage> images(imageCount);
    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, images.data());

    for (auto imageView : swapchainImageViews_) vkDestroyImageView(device_, imageView, NULL);
    swapchainImageViews_.resize(imageCount);
    for (int i = 0; i < imageCount; ++i) {
      VkImageViewCreateInfo imageViewInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      imageViewInfo.image = images[i];
      imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      imageViewInfo.format = swapchainInfo.imageFormat;
      imageViewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCreateImageView(device_, &imageViewInfo, NULL, &swapchainImageViews_[i]);
    }

    shouldRecreateFramebuffer_ = true;
  }

  void RecreateRenderPass() {
    // TODO: cache
    if (renderPass_) vkDestroyRenderPass(device_, renderPass_, NULL);

    std::vector<VkAttachmentDescription> attachments(1);
    attachments[0] = {};
    attachments[0].format = VK_FORMAT_B8G8R8A8_UNORM;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    std::vector<VkAttachmentReference> colorAttachments(1);
    colorAttachments[0] = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    std::vector<VkSubpassDescription> subpasses(1);
    subpasses[0] = {};
    subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses[0].colorAttachmentCount = colorAttachments.size();
    subpasses[0].pColorAttachments = colorAttachments.data();

    std::vector<VkSubpassDependency> dependencies(1);
    dependencies[0] = {};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    renderPassInfo.attachmentCount = attachments.size();
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = subpasses.size();
    renderPassInfo.pSubpasses = subpasses.data();
    renderPassInfo.dependencyCount = dependencies.size();
    renderPassInfo.pDependencies = dependencies.data();
    vkCreateRenderPass(device_, &renderPassInfo, NULL, &renderPass_);

    shouldRecreateUi_ = true;
    shouldRecreateFramebuffer_ = true;
    shouldRecreatePipelines_ = true;
  }

  void RecreateFramebuffer() {
    for (auto framebuffer : framebuffers_) vkDestroyFramebuffer(device_, framebuffer, NULL);
    framebuffers_.resize(swapchainImageViews_.size());

    for (int i = 0; i < swapchainImageViews_.size(); ++i) {
      std::vector<VkImageView> attachments = {
          // TODO: color/depth attachments
          swapchainImageViews_[i],
      };
      VkFramebufferCreateInfo framebufferInfo = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      framebufferInfo.renderPass = renderPass_;
      framebufferInfo.attachmentCount = attachments.size();
      framebufferInfo.pAttachments = attachments.data();
      framebufferInfo.width = swapchainWidth_;
      framebufferInfo.height = swapchainHeight_;
      framebufferInfo.layers = 1;
      vkCreateFramebuffer(device_, &framebufferInfo, NULL, &framebuffers_[i]);
    }
  }

  void RecreateUi() {
    viewer::UiCreateInfo uiInfo = {};
    uiInfo.instance = instance_;
    uiInfo.physicalDevice = physicalDevice_;
    uiInfo.device = device_;
    uiInfo.queueFamily = graphicsQueueFamily_;
    uiInfo.queue = graphicsQueue_;
    uiInfo.pipelineCache = pipelineCache_;
    uiInfo.descriptorPool = descriptorPool_;
    uiInfo.renderPass = renderPass_;
    uiInfo.subpass = 0;
    uiInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    viewer_.PrepareUi(uiInfo);
  }

  void RecreatePipelines() {
    if (colorTrianglePipeline_) vkDestroyPipeline(device_, colorTrianglePipeline_, NULL);

    // shader modules
    VkShaderModule colorVertModule;
    VkShaderModule colorFragModule;

    VkShaderModuleCreateInfo shaderModuleInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleInfo.codeSize = sizeof(color_vert);
    shaderModuleInfo.pCode = color_vert;
    vkCreateShaderModule(device_, &shaderModuleInfo, NULL, &colorVertModule);
    shaderModuleInfo.codeSize = sizeof(color_frag);
    shaderModuleInfo.pCode = color_frag;
    vkCreateShaderModule(device_, &shaderModuleInfo, NULL, &colorFragModule);

    std::vector<std::vector<VkPipelineShaderStageCreateInfo>> stages(1);
    std::vector<std::vector<VkVertexInputBindingDescription>> bindings(1);
    std::vector<std::vector<VkVertexInputAttributeDescription>> attributes(1);
    std::vector<VkPipelineVertexInputStateCreateInfo> vertexInputStates(1);
    std::vector<VkPipelineInputAssemblyStateCreateInfo> inputAssemblyStates(1);
    std::vector<VkPipelineViewportStateCreateInfo> viewportStates(1);
    std::vector<VkPipelineRasterizationStateCreateInfo> rasterizationStates(1);
    std::vector<VkPipelineMultisampleStateCreateInfo> multisampleStates(1);
    std::vector<VkPipelineDepthStencilStateCreateInfo> depthStencilStates(1);
    std::vector<std::vector<VkPipelineColorBlendAttachmentState>> colorBlendAttachments(1);
    std::vector<VkPipelineColorBlendStateCreateInfo> colorBlendStates(1);
    std::vector<VkPipelineDynamicStateCreateInfo> dynamicStates(1);
    std::vector<VkGraphicsPipelineCreateInfo> pipelineInfos(1);

    // color line pipeline
    stages[0].resize(2);
    stages[0][0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0][0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0][0].module = colorVertModule;
    stages[0][0].pName = "main";

    stages[0][1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0][1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[0][1].module = colorFragModule;
    stages[0][1].pName = "main";

    bindings[0].resize(1);
    bindings[0][0] = {0, sizeof(float) * 7, VK_VERTEX_INPUT_RATE_VERTEX};
    attributes[0].resize(2);
    attributes[0][0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 0};
    attributes[0][1] = {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(float) * 3};
    vertexInputStates[0] = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInputStates[0].vertexBindingDescriptionCount = bindings[0].size();
    vertexInputStates[0].pVertexBindingDescriptions = bindings[0].data();
    vertexInputStates[0].vertexAttributeDescriptionCount = attributes[0].size();
    vertexInputStates[0].pVertexAttributeDescriptions = attributes[0].data();

    inputAssemblyStates[0] = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssemblyStates[0].topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    viewportStates[0] = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportStates[0].viewportCount = 1;
    viewportStates[0].scissorCount = 1;

    rasterizationStates[0] = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizationStates[0].polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationStates[0].cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizationStates[0].frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationStates[0].lineWidth = 1.f;

    multisampleStates[0] = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampleStates[0].rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    depthStencilStates[0] = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencilStates[0].depthTestEnable = VK_TRUE;
    depthStencilStates[0].depthWriteEnable = VK_TRUE;
    depthStencilStates[0].depthCompareOp = VK_COMPARE_OP_LESS;

    colorBlendAttachments[0].resize(1);
    colorBlendAttachments[0][0].blendEnable = VK_TRUE;
    colorBlendAttachments[0][0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachments[0][0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachments[0][0].colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachments[0][0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachments[0][0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    colorBlendAttachments[0][0].alphaBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachments[0][0].colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    colorBlendStates[0] = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlendStates[0].attachmentCount = colorBlendAttachments[0].size();
    colorBlendStates[0].pAttachments = colorBlendAttachments[0].data();

    std::vector<VkDynamicState> dynamicStateList = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    dynamicStates[0] = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicStates[0].dynamicStateCount = dynamicStateList.size();
    dynamicStates[0].pDynamicStates = dynamicStateList.data();

    pipelineInfos[0] = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfos[0].stageCount = stages[0].size();
    pipelineInfos[0].pStages = stages[0].data();
    pipelineInfos[0].pVertexInputState = &vertexInputStates[0];
    pipelineInfos[0].pInputAssemblyState = &inputAssemblyStates[0];
    pipelineInfos[0].pViewportState = &viewportStates[0];
    pipelineInfos[0].pRasterizationState = &rasterizationStates[0];
    pipelineInfos[0].pMultisampleState = &multisampleStates[0];
    pipelineInfos[0].pDepthStencilState = &depthStencilStates[0];
    pipelineInfos[0].pColorBlendState = &colorBlendStates[0];
    pipelineInfos[0].pDynamicState = &dynamicStates[0];
    pipelineInfos[0].layout = graphicsPipelineLayout_;
    pipelineInfos[0].renderPass = renderPass_;
    pipelineInfos[0].subpass = 0;

    std::vector<VkPipeline> pipelines(pipelineInfos.size());
    vkCreateGraphicsPipelines(device_, pipelineCache_, pipelineInfos.size(), pipelineInfos.data(), NULL,
                              pipelines.data());

    colorTrianglePipeline_ = pipelines[0];

    vkDestroyShaderModule(device_, colorVertModule, NULL);
    vkDestroyShaderModule(device_, colorFragModule, NULL);
  }

  void ToggleDisplayMode() {
    switch (viewer_.displayMode()) {
      case viewer::DisplayMode::Windowed:
        viewer_.SetWindowedFullscreen();
        break;
      case viewer::DisplayMode::WindowedFullscreen:
        viewer_.SetWindowed();
        break;
    }
  }

  viewer::Viewer viewer_;

  Camera camera_;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue transferQueue_ = VK_NULL_HANDLE;
  VkQueue computeQueue_ = VK_NULL_HANDLE;
  VkQueue graphicsQueue_ = VK_NULL_HANDLE;
  uint32_t transferQueueFamily_ = 0;
  uint32_t computeQueueFamily_ = 0;
  uint32_t graphicsQueueFamily_ = 0;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  std::string deviceName_;

  VkCommandPool transferCommandPool_ = VK_NULL_HANDLE;
  VkCommandPool computeCommandPool_ = VK_NULL_HANDLE;
  VkCommandPool graphicsCommandPool_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
  VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;

  // primitives
  struct Buffer {
    VkBuffer buffer;
    VmaAllocation allocation;
    void* ptr;
  };

  struct Object {
    glm::mat4 model;
    Buffer vertexBuffer;  // (N, 7), float32
    Buffer indexBuffer;   // (M), uint32
    uint32_t indexCount;
  };
  Object triangle_ = {};

  // render pass
  VkRenderPass renderPass_ = VK_NULL_HANDLE;

  // layouts
  VkDescriptorSetLayout cameraSetLayout_ = VK_NULL_HANDLE;
  VkPipelineLayout graphicsPipelineLayout_ = VK_NULL_HANDLE;

  // descriptors and uniform buffers
  std::vector<VkDescriptorSet> cameraSets_;
  std::vector<Buffer> cameraBuffers_;

  // pipelines
  VkPipeline colorTrianglePipeline_ = VK_NULL_HANDLE;

  // swapchain
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  VkPresentModeKHR swapchainPresentMode_ = VK_PRESENT_MODE_FIFO_KHR;
  uint32_t swapchainWidth_ = 0;
  uint32_t swapchainHeight_ = 0;
  std::vector<VkImageView> swapchainImageViews_;

  // framebuffer
  std::vector<VkFramebuffer> framebuffers_;

  // transfer
  VkCommandBuffer transferCommandBuffer_ = VK_NULL_HANDLE;
  VkSemaphore transferSemaphore_ = VK_NULL_HANDLE;
  Buffer staging_ = {};

  // render
  std::vector<VkCommandBuffer> graphicsCommandBuffers_;
  std::vector<VkSemaphore> imageAcquiredSemaphores_;
  std::vector<VkSemaphore> renderFinishedSemaphores_;
  std::vector<VkFence> renderFinishedFences_;

  // dirty flags
  bool shouldRecreateSwapchain_ = false;
  bool shouldRecreateRenderPass_ = false;
  bool shouldRecreateFramebuffer_ = false;
  bool shouldRecreatePipelines_ = false;
  bool shouldRecreateUi_ = false;

  // counters
  int imageAcquireIndex_ = 0;
  int renderIndex_ = 0;
  uint64_t frameCounter_ = 0;
};

Engine::Engine() : impl_(std::make_shared<Impl>()) {}

Engine::~Engine() = default;

void Engine::Run() { impl_->Run(); }

}  // namespace vkgs
