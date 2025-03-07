#include <vkgs/engine/engine.h>

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <map>
#include <fstream>
#include <thread>
#include <chrono>

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"

#include <glm/gtc/type_ptr.hpp>

#include "vk_radix_sort.h"

#include "vkgs/viewer/viewer.h"
#include "vkgs/engine/load_ply.h"

#include <vkgs/scene/camera.h>
#include <vkgs/util/clock.h>

#include "generated/color_vert.h"
#include "generated/color_frag.h"
#include "generated/splat_vert.h"
#include "generated/splat_frag.h"
#include "generated/parse_ply_comp.h"
#include "generated/rank_comp.h"
#include "generated/inverse_index_comp.h"
#include "generated/projection_comp.h"

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

    VkPhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};

    VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bufferDeviceAddressFeatures.pNext = &timelineSemaphoreFeatures;

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
    std::vector<VkDeviceQueueCreateInfo> queueInfos(3);
    queueInfos[0] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfos[0].queueFamilyIndex = transferQueueFamily_;
    queueInfos[0].queueCount = 1;
    queueInfos[0].pQueuePriorities = &queuePriorities[0];

    queueInfos[1] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfos[1].queueFamilyIndex = computeQueueFamily_;
    queueInfos[1].queueCount = 1;
    queueInfos[1].pQueuePriorities = &queuePriorities[1];

    queueInfos[2] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfos[2].queueFamilyIndex = graphicsQueueFamily_;
    queueInfos[2].queueCount = 1;
    queueInfos[2].pQueuePriorities = &queuePriorities[2];

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
    commandBufferInfo.commandBufferCount = 2;
    graphicsCommandBuffers_.resize(commandBufferInfo.commandBufferCount);
    vkAllocateCommandBuffers(device_, &commandBufferInfo, graphicsCommandBuffers_.data());

    // layouts
    VkDescriptorSetLayoutBinding cameraBinding = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT};
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptorSetLayoutInfo.bindingCount = 1;
    descriptorSetLayoutInfo.pBindings = &cameraBinding;
    vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutInfo, NULL, &cameraSetLayout_);

    VkDescriptorSetLayoutBinding splatBinding = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT};
    descriptorSetLayoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptorSetLayoutInfo.bindingCount = 1;
    descriptorSetLayoutInfo.pBindings = &splatBinding;
    vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutInfo, NULL, &splatSetLayout_);

    std::vector<VkDescriptorSetLayout> setLayouts = {cameraSetLayout_, splatSetLayout_};
    VkPushConstantRange pushConstantRange = {VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4)};
    VkPipelineLayoutCreateInfo graphicsPipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    graphicsPipelineLayoutInfo.setLayoutCount = setLayouts.size();
    graphicsPipelineLayoutInfo.pSetLayouts = setLayouts.data();
    graphicsPipelineLayoutInfo.pushConstantRangeCount = 1;
    graphicsPipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    vkCreatePipelineLayout(device_, &graphicsPipelineLayoutInfo, NULL, &graphicsPipelineLayout_);

    // descriptors
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

    std::vector<VkDescriptorSetLayout> splatSetLayous = {
        splatSetLayout_,
        splatSetLayout_,
    };
    descriptorSetInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    descriptorSetInfo.descriptorPool = descriptorPool_;
    descriptorSetInfo.descriptorSetCount = splatSetLayous.size();
    descriptorSetInfo.pSetLayouts = splatSetLayous.data();
    splatSets_.resize(2);
    vkAllocateDescriptorSets(device_, &descriptorSetInfo, splatSets_.data());

    // gaussian splat
    PrepareGaussianSplat();

    // transfer
    std::vector<VkCommandBuffer> ownershipAcquiredCommandBuffers;
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

      commandBufferInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      commandBufferInfo.commandPool = graphicsCommandPool_;
      commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      commandBufferInfo.commandBufferCount = 2;
      ownershipAcquiredCommandBuffers.resize(commandBufferInfo.commandBufferCount);
      vkAllocateCommandBuffers(device_, &commandBufferInfo, ownershipAcquiredCommandBuffers.data());

      VkSemaphoreCreateInfo semaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      vkCreateSemaphore(device_, &semaphoreInfo, NULL, &transferSemaphore0_);
      vkCreateSemaphore(device_, &semaphoreInfo, NULL, &transferSemaphore1_);

      VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      vkCreateFence(device_, &fenceInfo, NULL, &transferFence_);
    }

    // triangle
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
      signalSemaphoreInfo.semaphore = transferSemaphore0_;
      signalSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
      VkSubmitInfo2 submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submitInfo.commandBufferInfoCount = 1;
      submitInfo.pCommandBufferInfos = &commandBufferInfo;
      submitInfo.signalSemaphoreInfoCount = 1;
      submitInfo.pSignalSemaphoreInfos = &signalSemaphoreInfo;
      vkQueueSubmit2(transferQueue_, 1, &submitInfo, transferFence_);

      // transfer ownership
      vkBeginCommandBuffer(ownershipAcquiredCommandBuffers[0], &commandBufferBeginInfo);

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
      vkCmdPipelineBarrier(ownershipAcquiredCommandBuffers[0], VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, NULL,
                           barriers.size(), barriers.data(), 0, NULL);

      vkEndCommandBuffer(ownershipAcquiredCommandBuffers[0]);

      VkSemaphoreSubmitInfo waitSemaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      waitSemaphoreInfo.semaphore = transferSemaphore0_;
      waitSemaphoreInfo.stageMask =
          VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
      commandBufferInfo.commandBuffer = ownershipAcquiredCommandBuffers[0];
      submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submitInfo.waitSemaphoreInfoCount = 1;
      submitInfo.pWaitSemaphoreInfos = &waitSemaphoreInfo;
      submitInfo.commandBufferInfoCount = 1;
      submitInfo.pCommandBufferInfos = &commandBufferInfo;
      vkQueueSubmit2(graphicsQueue_, 1, &submitInfo, NULL);
    }

    // buffers
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
    writes[1].pBufferInfo = &bufferInfos[1];
    vkUpdateDescriptorSets(device_, writes.size(), writes.data(), 0, NULL);

    frameInfos_.resize(2);
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

    for (const auto& frameInfo : frameInfos_) {
      if (frameInfo.computeQueryPool) vkDestroyQueryPool(device_, frameInfo.computeQueryPool, NULL);
      if (frameInfo.graphicsQueryPool) vkDestroyQueryPool(device_, frameInfo.graphicsQueryPool, NULL);
    }

    DestroyGaussianSplat();

    vkDestroyDescriptorSetLayout(device_, cameraSetLayout_, NULL);
    vkDestroyDescriptorSetLayout(device_, splatSetLayout_, NULL);
    vkDestroyPipelineLayout(device_, graphicsPipelineLayout_, NULL);
    if (colorTrianglePipeline_) vkDestroyPipeline(device_, colorTrianglePipeline_, NULL);
    if (splatPipeline_) vkDestroyPipeline(device_, splatPipeline_, NULL);

    for (auto semaphore : imageAcquiredSemaphores_) vkDestroySemaphore(device_, semaphore, NULL);
    for (auto semaphore : renderFinishedSemaphores_) vkDestroySemaphore(device_, semaphore, NULL);
    for (auto fence : renderFinishedFences_) vkDestroyFence(device_, fence, NULL);

    if (depthAttachment_.view) {
      vkDestroyImageView(device_, depthAttachment_.view, NULL);
      vmaDestroyImage(allocator_, depthAttachment_.image, depthAttachment_.allocation);
    }
    for (auto framebuffer : framebuffers_) vkDestroyFramebuffer(device_, framebuffer, NULL);

    vkDestroyRenderPass(device_, renderPass_, NULL);

    vkDestroyCommandPool(device_, transferCommandPool_, NULL);
    vkDestroyCommandPool(device_, computeCommandPool_, NULL);
    vkDestroyCommandPool(device_, graphicsCommandPool_, NULL);
    vkDestroyDescriptorPool(device_, descriptorPool_, NULL);

    vmaDestroyBuffer(allocator_, triangle_.vertexBuffer.buffer, triangle_.vertexBuffer.allocation);
    vmaDestroyBuffer(allocator_, triangle_.indexBuffer.buffer, triangle_.indexBuffer.allocation);

    vmaDestroyBuffer(allocator_, staging_.buffer, staging_.allocation);
    vkDestroySemaphore(device_, transferSemaphore0_, NULL);
    vkDestroySemaphore(device_, transferSemaphore1_, NULL);
    vkDestroyFence(device_, transferFence_, NULL);

    for (const auto& buffer : cameraBuffers_) vmaDestroyBuffer(allocator_, buffer.buffer, buffer.allocation);
    vmaDestroyAllocator(allocator_);

    vkDestroyDevice(device_, NULL);
    DestroyDebugUtilsMessengerEXT(instance_, messenger_, NULL);
    vkDestroyInstance(instance_, NULL);
  }

  void Run(const std::string& plyFilepath) {
    viewer_.PrepareWindow(instance_);

    RecreateSwapchain();
    RecreateRenderPass();
    RecreateFramebuffer();
    RecreateUi();
    shouldRecreateSwapchain_ = false;
    shouldRecreateRenderPass_ = false;
    shouldRecreateFramebuffer_ = false;
    shouldRecreateUi_ = false;

    if (!plyFilepath.empty()) {
      ParsePly(plyFilepath);
    }

    while (!viewer_.ShouldClose()) {
      viewer_.PollEvents();

      // draw ui
      viewer_.BeginUi();
      HandleEvents();
      DrawUi();
      viewer_.EndUi();

      DoGraphicsJobs();
    }

    vkDeviceWaitIdle(device_);

    vkDestroySwapchainKHR(device_, swapchain_, NULL);
    for (auto imageView : swapchainImageViews_) vkDestroyImageView(device_, imageView, NULL);
    viewer_.DestroyWindow();
  }

 private:
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

      if (shouldRecreateGraphicsPipelines_) {
        vkWaitForFences(device_, renderFinishedFences_.size(), renderFinishedFences_.data(), VK_TRUE, UINT64_MAX);
        RecreateGraphicsPipelines();
        shouldRecreateGraphicsPipelines_ = false;
      }

      camera_.SetWindowSize(swapchainWidth_, swapchainHeight_);

      auto& frameInfo = frameInfos_[renderIndex_];
      const GaussianSplatStorage& storage = gaussianStorages_[renderIndex_];
      const GaussianSplatQuad& quad = gaussianQuads_[renderIndex_];
      if (gaussianSplat_.pointCount > 0) {
        uint32_t pointCount = gaussianSplat_.pointCount;
        VkCommandBuffer cb0 = gaussianComputeCommandBuffers_[renderIndex_ * 2 + 0];
        VkCommandBuffer cb1 = gaussianComputeCommandBuffers_[renderIndex_ * 2 + 1];
        VkFence fence = gaussianComputeFences_[renderIndex_];

        vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);

        if (!frameInfo.computeQueryPool) {
          VkQueryPoolCreateInfo queryPoolInfo = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
          queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
          queryPoolInfo.queryCount = 5;
          vkCreateQueryPool(device_, &queryPoolInfo, NULL, &frameInfo.computeQueryPool);
        } else {
          std::vector<uint64_t> timestamps(5);
          vkGetQueryPoolResults(device_, frameInfo.computeQueryPool, 0, 5, sizeof(uint64_t) * timestamps.size(),
                                timestamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
          frameInfo.rankTime = timestamps[1] - timestamps[0];
          frameInfo.sortTime = timestamps[2] - timestamps[1];
          frameInfo.inverseTime = timestamps[3] - timestamps[2];
          frameInfo.projectionTime = timestamps[4] - timestamps[3];
          frameInfo.totalComputeTime = timestamps[4] - timestamps[0];

          std::memcpy(&frameInfo.visiblePointCount, storage.visiblePointCountStaging.ptr, sizeof(uint32_t));
        }

        // update uniforms
        GaussianCamera gaussianCamera = {};
        gaussianCamera.projection = camera_.ProjectionMatrix();
        gaussianCamera.view = camera_.ViewMatrix();
        gaussianCamera.camera_position = camera_.Eye();
        gaussianCamera.screen_size = {swapchainWidth_, swapchainHeight_};
        std::memcpy(gaussianCameraBuffers_[renderIndex_].ptr, &gaussianCamera, sizeof(gaussianCamera));

        // update descriptors
        std::vector<VkDescriptorBufferInfo> bufferInfos(11);
        bufferInfos[0] = {gaussianCameraBuffers_[renderIndex_].buffer, 0, sizeof(GaussianCamera)};
        bufferInfos[1] = {gaussianSplat_.position.buffer, 0, VK_WHOLE_SIZE};
        bufferInfos[2] = {gaussianSplat_.cov3d.buffer, 0, VK_WHOLE_SIZE};
        bufferInfos[3] = {gaussianSplat_.opacity.buffer, 0, VK_WHOLE_SIZE};
        bufferInfos[4] = {gaussianSplat_.sh.buffer, 0, VK_WHOLE_SIZE};
        bufferInfos[5] = {storage.visiblePointCount.buffer, 0, VK_WHOLE_SIZE};
        bufferInfos[6] = {storage.key.buffer, 0, VK_WHOLE_SIZE};
        bufferInfos[7] = {storage.value.buffer, 0, VK_WHOLE_SIZE};
        bufferInfos[8] = {storage.inverse.buffer, 0, VK_WHOLE_SIZE};
        bufferInfos[9] = {quad.indirect.buffer, 0, VK_WHOLE_SIZE};
        bufferInfos[10] = {quad.quad.buffer, 0, VK_WHOLE_SIZE};

        std::vector<VkWriteDescriptorSet> writes(11);
        // camera
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].dstSet = gaussianSets_[renderIndex_].camera;
        writes[0].dstBinding = 0;
        writes[0].dstArrayElement = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &bufferInfos[0];

        // gaussian
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[1].dstSet = gaussianSets_[renderIndex_].gaussian;
        writes[1].dstBinding = 0;
        writes[1].dstArrayElement = 0;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &bufferInfos[1];

        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[2].dstSet = gaussianSets_[renderIndex_].gaussian;
        writes[2].dstBinding = 1;
        writes[2].dstArrayElement = 0;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &bufferInfos[2];

        writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[3].dstSet = gaussianSets_[renderIndex_].gaussian;
        writes[3].dstBinding = 2;
        writes[3].dstArrayElement = 0;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo = &bufferInfos[3];

        writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[4].dstSet = gaussianSets_[renderIndex_].gaussian;
        writes[4].dstBinding = 3;
        writes[4].dstArrayElement = 0;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].pBufferInfo = &bufferInfos[4];

        // storage
        writes[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[5].dstSet = gaussianSets_[renderIndex_].quad;
        writes[5].dstBinding = 0;
        writes[5].dstArrayElement = 0;
        writes[5].descriptorCount = 1;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].pBufferInfo = &bufferInfos[5];

        writes[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[6].dstSet = gaussianSets_[renderIndex_].quad;
        writes[6].dstBinding = 1;
        writes[6].dstArrayElement = 0;
        writes[6].descriptorCount = 1;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[6].pBufferInfo = &bufferInfos[6];

        writes[7] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[7].dstSet = gaussianSets_[renderIndex_].quad;
        writes[7].dstBinding = 2;
        writes[7].dstArrayElement = 0;
        writes[7].descriptorCount = 1;
        writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[7].pBufferInfo = &bufferInfos[7];

        writes[8] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[8].dstSet = gaussianSets_[renderIndex_].quad;
        writes[8].dstBinding = 3;
        writes[8].dstArrayElement = 0;
        writes[8].descriptorCount = 1;
        writes[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[8].pBufferInfo = &bufferInfos[8];

        writes[9] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[9].dstSet = gaussianSets_[renderIndex_].quad;
        writes[9].dstBinding = 4;
        writes[9].dstArrayElement = 0;
        writes[9].descriptorCount = 1;
        writes[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[9].pBufferInfo = &bufferInfos[9];

        writes[10] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[10].dstSet = gaussianSets_[renderIndex_].quad;
        writes[10].dstBinding = 5;
        writes[10].dstArrayElement = 0;
        writes[10].descriptorCount = 1;
        writes[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[10].pBufferInfo = &bufferInfos[10];

        vkUpdateDescriptorSets(device_, writes.size(), writes.data(), 0, NULL);

        // commands
        VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb0, &beginInfo);

        vkCmdResetQueryPool(cb0, frameInfo.computeQueryPool, 0, 5);

        vkCmdWriteTimestamp(cb0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, frameInfo.computeQueryPool, 0);

        vkCmdFillBuffer(cb0, storage.visiblePointCount.buffer, 0, sizeof(uint32_t), 0);

        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                             0, NULL, 0, NULL);

        GaussianSplatPushConstant pushConstants = {};
        pushConstants.model = gaussianSplat_.model;
        pushConstants.pointCount = pointCount;

        std::vector<VkDescriptorSet> descriptors = {
            gaussianSets_[renderIndex_].camera,
            gaussianSets_[renderIndex_].gaussian,
            gaussianSets_[renderIndex_].quad,
        };
        vkCmdPushConstants(cb0, gaussianSplatPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdBindDescriptorSets(cb0, VK_PIPELINE_BIND_POINT_COMPUTE, gaussianSplatPipelineLayout_, 0,
                                descriptors.size(), descriptors.data(), 0, NULL);

        constexpr uint32_t localSize = 256;
        vkCmdBindPipeline(cb0, VK_PIPELINE_BIND_POINT_COMPUTE, rankPipeline_);
        vkCmdDispatch(cb0, (pointCount + localSize - 1) / localSize, 1, 1);

        vkCmdWriteTimestamp(cb0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, frameInfo.computeQueryPool, 1);

        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier,
                             0, NULL, 0, NULL);

        VkBufferCopy region = {};
        region.srcOffset = 0;
        region.dstOffset = 0;
        region.size = sizeof(uint32_t);
        vkCmdCopyBuffer(cb0, storage.visiblePointCount.buffer, storage.visiblePointCountStaging.buffer, 1, &region);

        vrdxCmdSortKeyValueIndirect(cb0, sorter_, pointCount, storage.visiblePointCount.buffer, 0, storage.key.buffer,
                                    0, storage.value.buffer, 0, storage.storage.buffer, 0, NULL, 0);

        vkCmdWriteTimestamp(cb0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, frameInfo.computeQueryPool, 2);

        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier,
                             0, NULL, 0, NULL);

        // sorter binds its own, so reset pipeline layout.
        vkCmdPushConstants(cb0, gaussianSplatPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdBindDescriptorSets(cb0, VK_PIPELINE_BIND_POINT_COMPUTE, gaussianSplatPipelineLayout_, 0,
                                descriptors.size(), descriptors.data(), 0, NULL);

        vkCmdFillBuffer(cb0, storage.inverse.buffer, 0, pointCount * sizeof(uint32_t), -1);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                             0, NULL, 0, NULL);

        vkCmdBindPipeline(cb0, VK_PIPELINE_BIND_POINT_COMPUTE, inverseIndexPipeline_);
        vkCmdDispatch(cb0, (pointCount + localSize - 1) / localSize, 1, 1);

        vkCmdWriteTimestamp(cb0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, frameInfo.computeQueryPool, 3);

        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &barrier, 0, NULL, 0, NULL);

        vkEndCommandBuffer(cb0);

        // submit preprocesses
        VkCommandBufferSubmitInfo commandBufferSubmitInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        commandBufferSubmitInfo.commandBuffer = cb0;
        VkSubmitInfo2 submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
        vkQueueSubmit2(computeQueue_, 1, &submitInfo, NULL);

        // projection after rendering finishes
        vkBeginCommandBuffer(cb1, &beginInfo);

        vkCmdPushConstants(cb1, gaussianSplatPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdBindDescriptorSets(cb1, VK_PIPELINE_BIND_POINT_COMPUTE, gaussianSplatPipelineLayout_, 0,
                                descriptors.size(), descriptors.data(), 0, NULL);

        vkCmdBindPipeline(cb1, VK_PIPELINE_BIND_POINT_COMPUTE, projectionPipeline_);
        vkCmdDispatch(cb1, (pointCount + localSize - 1) / localSize, 1, 1);

        vkCmdWriteTimestamp(cb1, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, frameInfo.computeQueryPool, 4);

        // barrier for next processing
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb1, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier,
                             0, NULL, 0, NULL);

        // transfer ownership
        std::vector<VkBufferMemoryBarrier> ownershipBarriers(2);
        ownershipBarriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        ownershipBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ownershipBarriers[0].srcQueueFamilyIndex = computeQueueFamily_;
        ownershipBarriers[0].dstQueueFamilyIndex = graphicsQueueFamily_;
        ownershipBarriers[0].buffer = quad.indirect.buffer;
        ownershipBarriers[0].offset = 0;
        ownershipBarriers[0].size = VK_WHOLE_SIZE;

        ownershipBarriers[1] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        ownershipBarriers[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ownershipBarriers[1].srcQueueFamilyIndex = computeQueueFamily_;
        ownershipBarriers[1].dstQueueFamilyIndex = graphicsQueueFamily_;
        ownershipBarriers[1].buffer = quad.quad.buffer;
        ownershipBarriers[1].offset = 0;
        ownershipBarriers[1].size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb1, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, 0, NULL, ownershipBarriers.size(),
                             ownershipBarriers.data(), 0, NULL);

        vkEndCommandBuffer(cb1);

        {
          VkSemaphoreSubmitInfo waitSemaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
          waitSemaphoreInfo.semaphore = gaussianTimelineSemaphores_[renderIndex_].semaphore;
          waitSemaphoreInfo.value = gaussianTimelineSemaphores_[renderIndex_].timeline;
          waitSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
          VkCommandBufferSubmitInfo commandBufferSubmitInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
          commandBufferSubmitInfo.commandBuffer = cb1;
          VkSemaphoreSubmitInfo signalSemaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
          signalSemaphoreInfo.semaphore = gaussianTimelineSemaphores_[renderIndex_].semaphore;
          signalSemaphoreInfo.value = gaussianTimelineSemaphores_[renderIndex_].timeline + 1;
          signalSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
          VkSubmitInfo2 submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
          submitInfo.waitSemaphoreInfoCount = 1;
          submitInfo.pWaitSemaphoreInfos = &waitSemaphoreInfo;
          submitInfo.commandBufferInfoCount = 1;
          submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
          submitInfo.signalSemaphoreInfoCount = 1;
          submitInfo.pSignalSemaphoreInfos = &signalSemaphoreInfo;
          vkResetFences(device_, 1, &fence);
          vkQueueSubmit2(computeQueue_, 1, &submitInfo, fence);
        }
      }

      VkCommandBuffer cb = graphicsCommandBuffers_[renderIndex_];
      VkSemaphore renderFinishedSemaphore = renderFinishedSemaphores_[renderIndex_];
      VkFence renderFinishedFence = renderFinishedFences_[renderIndex_];

      // wait for render finishes
      vkWaitForFences(device_, 1, &renderFinishedFence, VK_TRUE, UINT64_MAX);

      if (!frameInfo.graphicsQueryPool) {
        VkQueryPoolCreateInfo queryPoolInfo = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolInfo.queryCount = 2;
        vkCreateQueryPool(device_, &queryPoolInfo, NULL, &frameInfo.graphicsQueryPool);
      } else {
        std::vector<uint64_t> timestamps(2);
        vkGetQueryPoolResults(device_, frameInfo.graphicsQueryPool, 0, 2, sizeof(uint64_t) * timestamps.size(),
                              timestamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        frameInfo.drawTime = timestamps[1] - timestamps[0];
        frameInfo.totalGraphicsTime = timestamps[1] - timestamps[0];
      }

      // record draw commands
      VkCommandBufferBeginInfo commandBufferBeginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(cb, &commandBufferBeginInfo);

      vkCmdResetQueryPool(cb, frameInfo.graphicsQueryPool, 0, 2);

      // acquire ownership from compute queue
      if (gaussianSplat_.pointCount > 0) {
        VkBufferMemoryBarrier ownershipBarrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        ownershipBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        ownershipBarrier.srcQueueFamilyIndex = computeQueueFamily_;
        ownershipBarrier.dstQueueFamilyIndex = graphicsQueueFamily_;
        ownershipBarrier.buffer = quad.indirect.buffer;
        ownershipBarrier.offset = 0;
        ownershipBarrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb, 0, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, NULL, 1, &ownershipBarrier, 0, NULL);

        ownershipBarrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        ownershipBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        ownershipBarrier.srcQueueFamilyIndex = computeQueueFamily_;
        ownershipBarrier.dstQueueFamilyIndex = graphicsQueueFamily_;
        ownershipBarrier.buffer = quad.quad.buffer;
        ownershipBarrier.offset = 0;
        ownershipBarrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb, 0, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 1, &ownershipBarrier, 0, NULL);
      }

      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, frameInfo.graphicsQueryPool, 0);

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
      viewport.width = swapchainWidth_;
      viewport.height = swapchainHeight_;
      viewport.minDepth = 0.f;
      viewport.maxDepth = 1.f;
      vkCmdSetViewport(cb, 0, 1, &viewport);

      VkRect2D scissor;
      scissor.offset = {0, 0};
      scissor.extent = {static_cast<uint32_t>(swapchainWidth_), static_cast<uint32_t>(swapchainHeight_)};
      vkCmdSetScissor(cb, 0, 1, &scissor);

      // update uniform buffer
      UniformCamera camera;
      camera.projection = camera_.ProjectionMatrix();
      camera.view = camera_.ViewMatrix();
      camera.camera_position = camera_.Eye();
      camera.screen_size = {swapchainWidth_, swapchainHeight_};
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

      // splat
      if (gaussianSplat_.pointCount > 0) {
        std::vector<VkDescriptorBufferInfo> bufferInfos(1);
        bufferInfos[0] = {quad.quad.buffer, 0, VK_WHOLE_SIZE};
        std::vector<VkWriteDescriptorSet> writes(1);
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].dstSet = splatSets_[renderIndex_];
        writes[0].dstBinding = 0;
        writes[0].dstArrayElement = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &bufferInfos[0];
        vkUpdateDescriptorSets(device_, writes.size(), writes.data(), 0, NULL);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, splatPipeline_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout_, 1, 1,
                                &splatSets_[renderIndex_], 0, NULL);
        vkCmdBindIndexBuffer(cb, gaussianIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexedIndirect(cb, quad.indirect.buffer, 0, 1, 0);
      }

      viewer_.DrawUi(cb);

      vkCmdEndRenderPass(cb);

      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, frameInfo.graphicsQueryPool, 1);

      vkEndCommandBuffer(cb);

      std::vector<VkSemaphoreSubmitInfo> waitSemaphoreInfos;
      waitSemaphoreInfos.push_back({VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO});
      waitSemaphoreInfos.back().semaphore = imageAcquiredSemaphore;
      waitSemaphoreInfos.back().stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      if (gaussianSplat_.pointCount > 0) {
        waitSemaphoreInfos.push_back({VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO});
        waitSemaphoreInfos.back().semaphore = gaussianTimelineSemaphores_[renderIndex_].semaphore;
        waitSemaphoreInfos.back().value = gaussianTimelineSemaphores_[renderIndex_].timeline + 1;
        waitSemaphoreInfos.back().stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                                              VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
                                              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
      }

      VkCommandBufferSubmitInfo commandBufferInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
      commandBufferInfo.commandBuffer = cb;

      std::vector<VkSemaphoreSubmitInfo> signalSemaphoreInfos;
      signalSemaphoreInfos.push_back({VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO});
      signalSemaphoreInfos.back().semaphore = renderFinishedSemaphore;
      signalSemaphoreInfos.back().stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      if (gaussianSplat_.pointCount > 0) {
        signalSemaphoreInfos.push_back({VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO});
        signalSemaphoreInfos.back().semaphore = gaussianTimelineSemaphores_[renderIndex_].semaphore;
        signalSemaphoreInfos.back().value = gaussianTimelineSemaphores_[renderIndex_].timeline + 2;
        signalSemaphoreInfos.back().stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                                                VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
                                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
      }

      VkSubmitInfo2 submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submitInfo.waitSemaphoreInfoCount = waitSemaphoreInfos.size();
      submitInfo.pWaitSemaphoreInfos = waitSemaphoreInfos.data();
      submitInfo.commandBufferInfoCount = 1;
      submitInfo.pCommandBufferInfos = &commandBufferInfo;
      submitInfo.signalSemaphoreInfoCount = signalSemaphoreInfos.size();
      submitInfo.pSignalSemaphoreInfos = signalSemaphoreInfos.data();
      vkResetFences(device_, 1, &renderFinishedFence);
      vkQueueSubmit2(graphicsQueue_, 1, &submitInfo, renderFinishedFence);

      if (gaussianSplat_.pointCount > 0) {
        gaussianTimelineSemaphores_[renderIndex_].timeline += 2;
      }

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

      const auto& frameInfo = frameInfos_[renderIndex_];
      ImGui::Text("%u points", gaussianSplat_.pointCount);
      ImGui::Text("%u visible points (%5.2f %%)", frameInfo.visiblePointCount,
                  static_cast<float>(frameInfo.visiblePointCount) / gaussianSplat_.pointCount);

      // GPU memory
      VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
      vmaGetHeapBudgets(allocator_, budgets);
      VkPhysicalDeviceMemoryProperties properties;
      vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &properties);
      ImGui::Text("GPU memory");
      for (int i = 0; i < properties.memoryHeapCount; ++i) {
        ImGui::Text("heap %d: %u allocations, %llu bytes", i, budgets[i].statistics.allocationCount,
                    budgets[i].statistics.allocationBytes);
        ImGui::Text("heap %d: %u blocks, %llu bytes", i, budgets[i].statistics.allocationCount,
                    budgets[i].statistics.allocationBytes);
        ImGui::Text("heap %d: total %llu bytes, budget %llu bytes", i, budgets[i].usage, budgets[i].budget);
      }

      // draw time
      uint64_t totalTime = frameInfo.rankTime + frameInfo.sortTime + frameInfo.inverseTime + frameInfo.projectionTime +
                           frameInfo.drawTime;
      ImGui::Text("rank      : %6.3f ms (%5.2f %%)", static_cast<float>(frameInfo.rankTime) / 1e6,
                  static_cast<float>(frameInfo.rankTime) / totalTime * 100.f);
      ImGui::Text("sort      : %6.3f ms (%5.2f %%)", static_cast<float>(frameInfo.sortTime) / 1e6,
                  static_cast<float>(frameInfo.sortTime) / totalTime * 100.f);
      ImGui::Text("inverse   : %6.3f ms (%5.2f %%)", static_cast<float>(frameInfo.inverseTime) / 1e6,
                  static_cast<float>(frameInfo.inverseTime) / totalTime * 100.f);
      ImGui::Text("projection: %6.3f ms (%5.2f %%)", static_cast<float>(frameInfo.projectionTime) / 1e6,
                  static_cast<float>(frameInfo.projectionTime) / totalTime * 100.f);
      ImGui::Text("draw      : %6.3f ms (%5.2f %%)", static_cast<float>(frameInfo.drawTime) / 1e6,
                  static_cast<float>(frameInfo.drawTime) / totalTime * 100.f);
    }
    ImGui::End();
  }

  void RecreateSwapchain() {
    VkSurfaceKHR surface = viewer_.surface();

    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface, &surfaceCapabilities);

    VkSwapchainKHR oldSwapchain = swapchain_;
    VkSwapchainCreateInfoKHR swapchainInfo = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    swapchainInfo.surface = surface;
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

    std::vector<VkAttachmentDescription> attachments(2);
    attachments[0] = {};
    attachments[0].format = VK_FORMAT_B8G8R8A8_UNORM;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    attachments[1] = {};
    attachments[1].format = VK_FORMAT_D32_SFLOAT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::vector<std::vector<VkAttachmentReference>> colorAttachments(1);
    colorAttachments[0].resize(1);
    colorAttachments[0][0] = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    std::vector<VkAttachmentReference> depthAttachments(1);
    depthAttachments[0] = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    std::vector<VkSubpassDescription> subpasses(1);
    subpasses[0] = {};
    subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses[0].colorAttachmentCount = colorAttachments[0].size();
    subpasses[0].pColorAttachments = colorAttachments[0].data();
    subpasses[0].pDepthStencilAttachment = &depthAttachments[0];

    std::vector<VkSubpassDependency> dependencies(1);
    dependencies[0] = {};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

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
    shouldRecreateGraphicsPipelines_ = true;
  }

  void RecreateFramebuffer() {
    if (depthAttachment_.view) {
      vkDestroyImageView(device_, depthAttachment_.view, NULL);
      vmaDestroyImage(allocator_, depthAttachment_.image, depthAttachment_.allocation);
    }

    VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_D32_SFLOAT;
    imageInfo.extent = {swapchainWidth_, swapchainHeight_, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    vmaCreateImage(allocator_, &imageInfo, &allocationCreateInfo, &depthAttachment_.image, &depthAttachment_.allocation,
                   NULL);

    VkImageViewCreateInfo imageViewInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    imageViewInfo.image = depthAttachment_.image;
    imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewInfo.format = VK_FORMAT_D32_SFLOAT;
    imageViewInfo.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    vkCreateImageView(device_, &imageViewInfo, NULL, &depthAttachment_.view);

    for (auto framebuffer : framebuffers_) vkDestroyFramebuffer(device_, framebuffer, NULL);
    framebuffers_.resize(swapchainImageViews_.size());

    for (int i = 0; i < swapchainImageViews_.size(); ++i) {
      std::vector<VkImageView> attachments = {
          swapchainImageViews_[i],
          depthAttachment_.view,
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

  void RecreateGraphicsPipelines() {
    if (colorTrianglePipeline_) vkDestroyPipeline(device_, colorTrianglePipeline_, NULL);
    if (splatPipeline_) vkDestroyPipeline(device_, splatPipeline_, NULL);

    // shader modules
    VkShaderModule colorVertModule;
    VkShaderModule colorFragModule;
    VkShaderModule splatVertModule;
    VkShaderModule splatFragModule;

    VkShaderModuleCreateInfo shaderModuleInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleInfo.codeSize = sizeof(color_vert);
    shaderModuleInfo.pCode = color_vert;
    vkCreateShaderModule(device_, &shaderModuleInfo, NULL, &colorVertModule);
    shaderModuleInfo.codeSize = sizeof(color_frag);
    shaderModuleInfo.pCode = color_frag;
    vkCreateShaderModule(device_, &shaderModuleInfo, NULL, &colorFragModule);

    shaderModuleInfo.codeSize = sizeof(splat_vert);
    shaderModuleInfo.pCode = splat_vert;
    vkCreateShaderModule(device_, &shaderModuleInfo, NULL, &splatVertModule);
    shaderModuleInfo.codeSize = sizeof(splat_frag);
    shaderModuleInfo.pCode = splat_frag;
    vkCreateShaderModule(device_, &shaderModuleInfo, NULL, &splatFragModule);

    std::vector<std::vector<VkPipelineShaderStageCreateInfo>> stages(2);
    std::vector<std::vector<VkVertexInputBindingDescription>> bindings(2);
    std::vector<std::vector<VkVertexInputAttributeDescription>> attributes(2);
    std::vector<VkPipelineVertexInputStateCreateInfo> vertexInputStates(2);
    std::vector<VkPipelineInputAssemblyStateCreateInfo> inputAssemblyStates(2);
    std::vector<VkPipelineViewportStateCreateInfo> viewportStates(2);
    std::vector<VkPipelineRasterizationStateCreateInfo> rasterizationStates(2);
    std::vector<VkPipelineMultisampleStateCreateInfo> multisampleStates(2);
    std::vector<VkPipelineDepthStencilStateCreateInfo> depthStencilStates(2);
    std::vector<std::vector<VkPipelineColorBlendAttachmentState>> colorBlendAttachments(2);
    std::vector<VkPipelineColorBlendStateCreateInfo> colorBlendStates(2);
    std::vector<VkPipelineDynamicStateCreateInfo> dynamicStates(2);
    std::vector<VkGraphicsPipelineCreateInfo> pipelineInfos(2);

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
    colorBlendAttachments[0][0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
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

    // splat pipeline
    stages[1].resize(2);
    stages[1][0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1][0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[1][0].module = splatVertModule;
    stages[1][0].pName = "main";

    stages[1][1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1][1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1][1].module = splatFragModule;
    stages[1][1].pName = "main";

    vertexInputStates[1] = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

    inputAssemblyStates[1] = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssemblyStates[1].topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    viewportStates[1] = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportStates[1].viewportCount = 1;
    viewportStates[1].scissorCount = 1;

    rasterizationStates[1] = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizationStates[1].polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationStates[1].cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizationStates[1].frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationStates[1].lineWidth = 1.f;

    multisampleStates[1] = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampleStates[1].rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    depthStencilStates[1] = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencilStates[1].depthTestEnable = VK_TRUE;
    depthStencilStates[1].depthWriteEnable = VK_FALSE;
    depthStencilStates[1].depthCompareOp = VK_COMPARE_OP_LESS;

    colorBlendAttachments[1].resize(1);
    colorBlendAttachments[1][0].blendEnable = VK_TRUE;
    colorBlendAttachments[1][0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachments[1][0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachments[1][0].colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachments[1][0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachments[1][0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachments[1][0].alphaBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachments[1][0].colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    colorBlendStates[1] = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlendStates[1].attachmentCount = colorBlendAttachments[1].size();
    colorBlendStates[1].pAttachments = colorBlendAttachments[1].data();

    dynamicStates[1] = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicStates[1].dynamicStateCount = dynamicStateList.size();
    dynamicStates[1].pDynamicStates = dynamicStateList.data();

    pipelineInfos[1] = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfos[1].stageCount = stages[1].size();
    pipelineInfos[1].pStages = stages[1].data();
    pipelineInfos[1].pVertexInputState = &vertexInputStates[1];
    pipelineInfos[1].pInputAssemblyState = &inputAssemblyStates[1];
    pipelineInfos[1].pViewportState = &viewportStates[1];
    pipelineInfos[1].pRasterizationState = &rasterizationStates[1];
    pipelineInfos[1].pMultisampleState = &multisampleStates[1];
    pipelineInfos[1].pDepthStencilState = &depthStencilStates[1];
    pipelineInfos[1].pColorBlendState = &colorBlendStates[1];
    pipelineInfos[1].pDynamicState = &dynamicStates[1];
    pipelineInfos[1].layout = graphicsPipelineLayout_;
    pipelineInfos[1].renderPass = renderPass_;
    pipelineInfos[1].subpass = 0;

    std::vector<VkPipeline> pipelines(pipelineInfos.size());
    vkCreateGraphicsPipelines(device_, pipelineCache_, pipelineInfos.size(), pipelineInfos.data(), NULL,
                              pipelines.data());

    colorTrianglePipeline_ = pipelines[0];
    splatPipeline_ = pipelines[1];

    vkDestroyShaderModule(device_, colorVertModule, NULL);
    vkDestroyShaderModule(device_, colorFragModule, NULL);
    vkDestroyShaderModule(device_, splatVertModule, NULL);
    vkDestroyShaderModule(device_, splatFragModule, NULL);
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

  void PrepareGaussianSplat() {
    // set layouts
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
    };
    VkDescriptorSetLayoutCreateInfo setLayoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setLayoutInfo.bindingCount = bindings.size();
    setLayoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device_, &setLayoutInfo, NULL, &gaussianCameraSetLayout_);

    bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
    };
    setLayoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setLayoutInfo.bindingCount = bindings.size();
    setLayoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device_, &setLayoutInfo, NULL, &gaussianSetLayout_);

    bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
    };
    setLayoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setLayoutInfo.bindingCount = bindings.size();
    setLayoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device_, &setLayoutInfo, NULL, &gaussianQuadSetLayout_);

    bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
    };
    setLayoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setLayoutInfo.bindingCount = bindings.size();
    setLayoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device_, &setLayoutInfo, NULL, &gaussianPlySetLayout_);

    // pipeline layout
    std::vector<VkDescriptorSetLayout> setLayouts = {
        gaussianCameraSetLayout_,
        gaussianSetLayout_,
        gaussianQuadSetLayout_,
        gaussianPlySetLayout_,
    };
    VkPushConstantRange pushConstantRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GaussianSplatPushConstant)};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = setLayouts.size();
    pipelineLayoutInfo.pSetLayouts = setLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    vkCreatePipelineLayout(device_, &pipelineLayoutInfo, NULL, &gaussianSplatPipelineLayout_);

    // pipelines
    VkShaderModule parsePlyModule;
    VkShaderModule rankModule;
    VkShaderModule inverseIndexModule;
    VkShaderModule projectionModule;

    VkShaderModuleCreateInfo shaderModuleInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleInfo.codeSize = sizeof(parse_ply_comp);
    shaderModuleInfo.pCode = parse_ply_comp;
    vkCreateShaderModule(device_, &shaderModuleInfo, NULL, &parsePlyModule);

    shaderModuleInfo.codeSize = sizeof(rank_comp);
    shaderModuleInfo.pCode = rank_comp;
    vkCreateShaderModule(device_, &shaderModuleInfo, NULL, &rankModule);

    shaderModuleInfo.codeSize = sizeof(inverse_index_comp);
    shaderModuleInfo.pCode = inverse_index_comp;
    vkCreateShaderModule(device_, &shaderModuleInfo, NULL, &inverseIndexModule);

    shaderModuleInfo.codeSize = sizeof(projection_comp);
    shaderModuleInfo.pCode = projection_comp;
    vkCreateShaderModule(device_, &shaderModuleInfo, NULL, &projectionModule);

    std::vector<VkComputePipelineCreateInfo> pipelineInfos(4);
    pipelineInfos[0] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfos[0].stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfos[0].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfos[0].stage.module = parsePlyModule;
    pipelineInfos[0].stage.pName = "main";
    pipelineInfos[0].layout = gaussianSplatPipelineLayout_;

    pipelineInfos[1] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfos[1].stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfos[1].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfos[1].stage.module = rankModule;
    pipelineInfos[1].stage.pName = "main";
    pipelineInfos[1].layout = gaussianSplatPipelineLayout_;

    pipelineInfos[2] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfos[2].stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfos[2].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfos[2].stage.module = inverseIndexModule;
    pipelineInfos[2].stage.pName = "main";
    pipelineInfos[2].layout = gaussianSplatPipelineLayout_;

    pipelineInfos[3] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfos[3].stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfos[3].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfos[3].stage.module = projectionModule;
    pipelineInfos[3].stage.pName = "main";
    pipelineInfos[3].layout = gaussianSplatPipelineLayout_;

    std::vector<VkPipeline> pipelines(pipelineInfos.size());
    vkCreateComputePipelines(device_, pipelineCache_, pipelineInfos.size(), pipelineInfos.data(), NULL,
                             pipelines.data());

    parsePlyPipeline_ = pipelines[0];
    rankPipeline_ = pipelines[1];
    inverseIndexPipeline_ = pipelines[2];
    projectionPipeline_ = pipelines[3];

    vkDestroyShaderModule(device_, parsePlyModule, NULL);
    vkDestroyShaderModule(device_, rankModule, NULL);
    vkDestroyShaderModule(device_, inverseIndexModule, NULL);
    vkDestroyShaderModule(device_, projectionModule, NULL);

    VrdxSorterCreateInfo sorterInfo = {};
    sorterInfo.device = device_;
    sorterInfo.physicalDevice = physicalDevice_;
    sorterInfo.pipelineCache = pipelineCache_;
    vrdxCreateSorter(&sorterInfo, &sorter_);

    // descriptor sets
    gaussianSets_.resize(2);
    for (int i = 0; i < gaussianSets_.size(); ++i) {
      std::vector<VkDescriptorSetLayout> setLayouts = {
          gaussianCameraSetLayout_,
          gaussianSetLayout_,
          gaussianQuadSetLayout_,
          gaussianPlySetLayout_,
      };
      VkDescriptorSetAllocateInfo descriptorSetInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      descriptorSetInfo.descriptorPool = descriptorPool_;
      descriptorSetInfo.descriptorSetCount = setLayouts.size();
      descriptorSetInfo.pSetLayouts = setLayouts.data();
      std::vector<VkDescriptorSet> descriptors(setLayouts.size());
      vkAllocateDescriptorSets(device_, &descriptorSetInfo, descriptors.data());

      gaussianSets_[i].camera = descriptors[0];
      gaussianSets_[i].gaussian = descriptors[1];
      gaussianSets_[i].quad = descriptors[2];
      gaussianSets_[i].ply = descriptors[3];
    }

    // uniform buffers
    gaussianCameraBuffers_.resize(2);
    for (int i = 0; i < gaussianCameraBuffers_.size(); ++i) {
      VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bufferInfo.size = sizeof(GaussianCamera);
      bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
      VmaAllocationCreateInfo allocationCreateInfo = {};
      allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
      allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
      VmaAllocationInfo allocationInfo;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &gaussianCameraBuffers_[i].buffer,
                      &gaussianCameraBuffers_[i].allocation, &allocationInfo);
      gaussianCameraBuffers_[i].ptr = allocationInfo.pMappedData;
    }

    // command buffer
    VkCommandBufferAllocateInfo commandBufferInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandBufferInfo.commandPool = computeCommandPool_;
    commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferInfo.commandBufferCount = 4;
    gaussianComputeCommandBuffers_.resize(commandBufferInfo.commandBufferCount);
    vkAllocateCommandBuffers(device_, &commandBufferInfo, gaussianComputeCommandBuffers_.data());

    // synchronizations
    VkSemaphoreCreateInfo semaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(device_, &semaphoreInfo, NULL, &gaussianPlyTransferSemaphore_);

    gaussianComputeFences_.resize(2);
    for (int i = 0; i < 2; ++i) {
      VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
      vkCreateFence(device_, &fenceInfo, NULL, &gaussianComputeFences_[i]);
    }

    gaussianTimelineSemaphores_.resize(2);
    for (auto& semaphore : gaussianTimelineSemaphores_) {
      semaphore.timeline = 0;

      VkSemaphoreTypeCreateInfo timelineSemaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
      timelineSemaphoreInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
      timelineSemaphoreInfo.initialValue = semaphore.timeline;
      VkSemaphoreCreateInfo semaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      semaphoreInfo.pNext = &timelineSemaphoreInfo;
      vkCreateSemaphore(device_, &semaphoreInfo, NULL, &semaphore.semaphore);
    }
  }

  void DestroyGaussianSplat() {
    vrdxDestroySorter(sorter_);

    for (auto& buffer : gaussianCameraBuffers_) vmaDestroyBuffer(allocator_, buffer.buffer, buffer.allocation);

    vkDestroyDescriptorSetLayout(device_, gaussianCameraSetLayout_, NULL);
    vkDestroyDescriptorSetLayout(device_, gaussianSetLayout_, NULL);
    vkDestroyDescriptorSetLayout(device_, gaussianQuadSetLayout_, NULL);
    vkDestroyDescriptorSetLayout(device_, gaussianPlySetLayout_, NULL);
    vkDestroyPipelineLayout(device_, gaussianSplatPipelineLayout_, NULL);
    vkDestroyPipeline(device_, parsePlyPipeline_, NULL);
    vkDestroyPipeline(device_, rankPipeline_, NULL);
    vkDestroyPipeline(device_, inverseIndexPipeline_, NULL);
    vkDestroyPipeline(device_, projectionPipeline_, NULL);

    if (gaussianPly_.plyBuffer.buffer) {
      vmaDestroyBuffer(allocator_, gaussianPly_.plyBuffer.buffer, gaussianPly_.plyBuffer.allocation);
    }

    if (gaussianSplat_.position.buffer)
      vmaDestroyBuffer(allocator_, gaussianSplat_.position.buffer, gaussianSplat_.position.allocation);
    if (gaussianSplat_.cov3d.buffer)
      vmaDestroyBuffer(allocator_, gaussianSplat_.cov3d.buffer, gaussianSplat_.cov3d.allocation);
    if (gaussianSplat_.opacity.buffer)
      vmaDestroyBuffer(allocator_, gaussianSplat_.opacity.buffer, gaussianSplat_.opacity.allocation);
    if (gaussianSplat_.sh.buffer) vmaDestroyBuffer(allocator_, gaussianSplat_.sh.buffer, gaussianSplat_.sh.allocation);

    for (const auto& storage : gaussianStorages_) {
      if (storage.key.buffer) vmaDestroyBuffer(allocator_, storage.key.buffer, storage.key.allocation);
      if (storage.value.buffer) vmaDestroyBuffer(allocator_, storage.value.buffer, storage.value.allocation);
      if (storage.storage.buffer) vmaDestroyBuffer(allocator_, storage.storage.buffer, storage.storage.allocation);
      if (storage.inverse.buffer) vmaDestroyBuffer(allocator_, storage.inverse.buffer, storage.inverse.allocation);
      if (storage.visiblePointCount.buffer)
        vmaDestroyBuffer(allocator_, storage.visiblePointCount.buffer, storage.visiblePointCount.allocation);
      if (storage.visiblePointCountStaging.buffer)
        vmaDestroyBuffer(allocator_, storage.visiblePointCountStaging.buffer,
                         storage.visiblePointCountStaging.allocation);
    }

    for (const auto& quad : gaussianQuads_) {
      if (quad.indirect.buffer) vmaDestroyBuffer(allocator_, quad.indirect.buffer, quad.indirect.allocation);
      if (quad.quad.buffer) vmaDestroyBuffer(allocator_, quad.quad.buffer, quad.quad.allocation);
    }
    if (gaussianIndexBuffer.buffer)
      vmaDestroyBuffer(allocator_, gaussianIndexBuffer.buffer, gaussianIndexBuffer.allocation);

    vkDestroySemaphore(device_, gaussianPlyTransferSemaphore_, NULL);
    for (auto fence : gaussianComputeFences_) vkDestroyFence(device_, fence, NULL);
    for (auto& semaphore : gaussianTimelineSemaphores_) vkDestroySemaphore(device_, semaphore.semaphore, NULL);
  }

  void ParsePly(const std::string& plyFilepath) {
    std::cout << "Loading " << plyFilepath << std::endl;

    auto plyBuffer = LoadPly(plyFilepath);

    std::cout << "Loaded" << std::endl;

    auto pointCount = plyBuffer.pointCount;

    // allocate gaussian buffers
    {
      gaussianSplat_.pointCount = pointCount;
      gaussianSplat_.model = glm::mat4{1.f};

      if (gaussianSplat_.position.buffer)
        vmaDestroyBuffer(allocator_, gaussianSplat_.position.buffer, gaussianSplat_.position.allocation);
      if (gaussianSplat_.cov3d.buffer)
        vmaDestroyBuffer(allocator_, gaussianSplat_.cov3d.buffer, gaussianSplat_.cov3d.allocation);
      if (gaussianSplat_.opacity.buffer)
        vmaDestroyBuffer(allocator_, gaussianSplat_.opacity.buffer, gaussianSplat_.opacity.allocation);
      if (gaussianSplat_.sh.buffer)
        vmaDestroyBuffer(allocator_, gaussianSplat_.sh.buffer, gaussianSplat_.sh.allocation);

      VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bufferInfo.size = pointCount * 3 * sizeof(float);
      bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      VmaAllocationCreateInfo allocationCreateInfo = {};
      allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &gaussianSplat_.position.buffer,
                      &gaussianSplat_.position.allocation, NULL);
      bufferInfo.size = pointCount * 6 * sizeof(float);
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &gaussianSplat_.cov3d.buffer,
                      &gaussianSplat_.cov3d.allocation, NULL);
      bufferInfo.size = pointCount * 1 * sizeof(float);
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &gaussianSplat_.opacity.buffer,
                      &gaussianSplat_.opacity.allocation, NULL);
      bufferInfo.size = pointCount * 3 * 16 * sizeof(uint16_t);
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &gaussianSplat_.sh.buffer,
                      &gaussianSplat_.sh.allocation, NULL);
    }

    // allocate storage buffers
    gaussianStorages_.resize(2);
    for (auto& storage : gaussianStorages_) {
      if (storage.key.buffer) vmaDestroyBuffer(allocator_, storage.key.buffer, storage.key.allocation);
      if (storage.value.buffer) vmaDestroyBuffer(allocator_, storage.value.buffer, storage.value.allocation);
      if (storage.storage.buffer) vmaDestroyBuffer(allocator_, storage.storage.buffer, storage.storage.allocation);
      if (storage.storage.buffer) vmaDestroyBuffer(allocator_, storage.inverse.buffer, storage.inverse.allocation);
      if (storage.visiblePointCount.buffer)
        vmaDestroyBuffer(allocator_, storage.visiblePointCount.buffer, storage.visiblePointCount.allocation);
      if (storage.visiblePointCountStaging.buffer)
        vmaDestroyBuffer(allocator_, storage.visiblePointCountStaging.buffer,
                         storage.visiblePointCountStaging.allocation);

      VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bufferInfo.size = pointCount * sizeof(uint32_t);
      bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      VmaAllocationCreateInfo allocationCreateInfo = {};
      allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &storage.key.buffer, &storage.key.allocation,
                      NULL);
      bufferInfo.size = pointCount * sizeof(uint32_t);
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &storage.value.buffer, &storage.value.allocation,
                      NULL);
      bufferInfo.size = pointCount * sizeof(uint32_t);
      bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &storage.inverse.buffer,
                      &storage.inverse.allocation, NULL);

      VrdxSorterStorageRequirements requirements;
      vrdxGetSorterStorageRequirements(sorter_, pointCount, &requirements);
      bufferInfo.size = requirements.size;
      bufferInfo.usage = requirements.usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &storage.storage.buffer,
                      &storage.storage.allocation, NULL);

      bufferInfo.size = sizeof(uint32_t);
      bufferInfo.usage =
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &storage.visiblePointCount.buffer,
                      &storage.visiblePointCount.allocation, NULL);

      bufferInfo.size = sizeof(uint32_t);
      bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
      allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
      VmaAllocationInfo allocationInfo;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &storage.visiblePointCountStaging.buffer,
                      &storage.visiblePointCountStaging.allocation, &allocationInfo);
      storage.visiblePointCountStaging.ptr = allocationInfo.pMappedData;
    }

    // allocate quads
    gaussianQuads_.resize(2);
    for (auto& quad : gaussianQuads_) {
      if (quad.indirect.buffer) vmaDestroyBuffer(allocator_, quad.indirect.buffer, quad.indirect.allocation);
      if (quad.quad.buffer) vmaDestroyBuffer(allocator_, quad.quad.buffer, quad.quad.allocation);

      VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bufferInfo.size = sizeof(VkDrawIndexedIndirectCommand);
      bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
      VmaAllocationCreateInfo allocationCreateInfo = {};
      allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &quad.indirect.buffer, &quad.indirect.allocation,
                      NULL);

      bufferInfo.size = pointCount * 12 * sizeof(float);
      bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &quad.quad.buffer, &quad.quad.allocation, NULL);
    }

    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = pointCount * 6 * sizeof(uint32_t);
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    VmaAllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &gaussianIndexBuffer.buffer,
                    &gaussianIndexBuffer.allocation, NULL);

    // to staging
    VkDeviceSize offsetSize = plyBuffer.plyOffsets.size() * sizeof(plyBuffer.plyOffsets[0]);
    VkDeviceSize bufferSize = plyBuffer.buffer.size() * sizeof(plyBuffer.buffer[0]);
    VkDeviceSize size = offsetSize + bufferSize;
    {
      VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bufferInfo.size = size;
      bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      VmaAllocationCreateInfo allocationCreateInfo = {};
      allocationCreateInfo.flags =
          VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
      allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
      VmaAllocationInfo allocationInfo;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &gaussianPly_.staging.buffer,
                      &gaussianPly_.staging.allocation, &allocationInfo);
      gaussianPly_.staging.ptr = allocationInfo.pMappedData;

      bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      allocationCreateInfo = {};
      allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
      vmaCreateBuffer(allocator_, &bufferInfo, &allocationCreateInfo, &gaussianPly_.plyBuffer.buffer,
                      &gaussianPly_.plyBuffer.allocation, NULL);

      std::cout << "to staging buffer, " << size << " bytes" << std::endl;
      std::memcpy(gaussianPly_.staging.ptr, plyBuffer.plyOffsets.data(), offsetSize);
      std::memcpy(reinterpret_cast<uint8_t*>(gaussianPly_.staging.ptr) + offsetSize, plyBuffer.buffer.data(),
                  bufferSize);
      std::cout << "to staging buffer done" << std::endl;
    }

    // transfer and release ownership
    {
      vkWaitForFences(device_, 1, &transferFence_, VK_TRUE, UINT64_MAX);

      VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(transferCommandBuffer_, &beginInfo);

      VkBufferCopy region = {};
      region.srcOffset = 0;
      region.dstOffset = 0;
      region.size = size;
      vkCmdCopyBuffer(transferCommandBuffer_, gaussianPly_.staging.buffer, gaussianPly_.plyBuffer.buffer, 1, &region);

      VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.srcQueueFamilyIndex = transferQueueFamily_;
      barrier.dstQueueFamilyIndex = computeQueueFamily_;
      barrier.buffer = gaussianPly_.plyBuffer.buffer;
      barrier.offset = 0;
      barrier.size = size;
      vkCmdPipelineBarrier(transferCommandBuffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, NULL, 1, &barrier, 0, NULL);

      vkEndCommandBuffer(transferCommandBuffer_);

      VkCommandBufferSubmitInfo commandBufferSubmitInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
      commandBufferSubmitInfo.commandBuffer = transferCommandBuffer_;
      VkSemaphoreSubmitInfo signalSemaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      signalSemaphoreInfo.semaphore = gaussianPlyTransferSemaphore_;
      signalSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
      VkSubmitInfo2 submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submitInfo.commandBufferInfoCount = 1;
      submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
      submitInfo.signalSemaphoreInfoCount = 1;
      submitInfo.pSignalSemaphoreInfos = &signalSemaphoreInfo;
      vkResetFences(device_, 1, &transferFence_);
      vkQueueSubmit2(transferQueue_, 1, &submitInfo, transferFence_);
    }

    // acquire ownership and parse ply
    {
      VkCommandBuffer cb = gaussianComputeCommandBuffers_[renderIndex_];
      VkFence fence = gaussianComputeFences_[renderIndex_];

      vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);

      // update descriptors
      VkDescriptorSet gaussianSet = gaussianSets_[renderIndex_].gaussian;
      VkDescriptorSet plySet = gaussianSets_[renderIndex_].ply;

      std::vector<VkDescriptorBufferInfo> bufferInfos(5);
      bufferInfos[0] = {gaussianSplat_.position.buffer, 0, VK_WHOLE_SIZE};
      bufferInfos[1] = {gaussianSplat_.cov3d.buffer, 0, VK_WHOLE_SIZE};
      bufferInfos[2] = {gaussianSplat_.opacity.buffer, 0, VK_WHOLE_SIZE};
      bufferInfos[3] = {gaussianSplat_.sh.buffer, 0, VK_WHOLE_SIZE};
      bufferInfos[4] = {gaussianPly_.plyBuffer.buffer, 0, VK_WHOLE_SIZE};

      std::vector<VkWriteDescriptorSet> writes(5);
      writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[0].dstSet = gaussianSet;
      writes[0].dstBinding = 0;
      writes[0].dstArrayElement = 0;
      writes[0].descriptorCount = 1;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[0].pBufferInfo = &bufferInfos[0];

      writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[1].dstSet = gaussianSet;
      writes[1].dstBinding = 1;
      writes[1].dstArrayElement = 0;
      writes[1].descriptorCount = 1;
      writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[1].pBufferInfo = &bufferInfos[1];

      writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[2].dstSet = gaussianSet;
      writes[2].dstBinding = 2;
      writes[2].dstArrayElement = 0;
      writes[2].descriptorCount = 1;
      writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[2].pBufferInfo = &bufferInfos[2];

      writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[3].dstSet = gaussianSet;
      writes[3].dstBinding = 3;
      writes[3].dstArrayElement = 0;
      writes[3].descriptorCount = 1;
      writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[3].pBufferInfo = &bufferInfos[3];

      writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[4].dstSet = plySet;
      writes[4].dstBinding = 0;
      writes[4].dstArrayElement = 0;
      writes[4].descriptorCount = 1;
      writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[4].pBufferInfo = &bufferInfos[4];
      vkUpdateDescriptorSets(device_, writes.size(), writes.data(), 0, NULL);

      VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(cb, &beginInfo);

      VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barrier.srcQueueFamilyIndex = transferQueueFamily_;
      barrier.dstQueueFamilyIndex = computeQueueFamily_;
      barrier.buffer = gaussianPly_.plyBuffer.buffer;
      barrier.offset = 0;
      barrier.size = size;
      vkCmdPipelineBarrier(cb, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &barrier, 0, NULL);

      // parse ply
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, parsePlyPipeline_);
      vkCmdPushConstants(cb, gaussianSplatPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(glm::mat4),
                         sizeof(uint32_t), &pointCount);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, gaussianSplatPipelineLayout_, 1, 1, &gaussianSet, 0,
                              NULL);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, gaussianSplatPipelineLayout_, 3, 1, &plySet, 0, NULL);

      constexpr uint32_t localSize = 256;
      uint32_t groupSize = (pointCount + localSize - 1) / localSize;
      vkCmdDispatch(cb, groupSize, 1, 1);

      vkEndCommandBuffer(cb);

      VkSemaphoreSubmitInfo waitSemaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      waitSemaphoreInfo.semaphore = gaussianPlyTransferSemaphore_;
      waitSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      VkCommandBufferSubmitInfo commandBufferSubmitInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
      commandBufferSubmitInfo.commandBuffer = cb;
      VkSubmitInfo2 submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submitInfo.waitSemaphoreInfoCount = 1;
      submitInfo.pWaitSemaphoreInfos = &waitSemaphoreInfo;
      submitInfo.commandBufferInfoCount = 1;
      submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
      vkResetFences(device_, 1, &fence);
      vkQueueSubmit2(computeQueue_, 1, &submitInfo, fence);
    }

    // fill index buffer
    {
      std::vector<uint32_t> indexBuffer;
      for (int i = 0; i < pointCount; ++i) {
        indexBuffer.push_back(4 * i + 0);
        indexBuffer.push_back(4 * i + 1);
        indexBuffer.push_back(4 * i + 2);
        indexBuffer.push_back(4 * i + 2);
        indexBuffer.push_back(4 * i + 1);
        indexBuffer.push_back(4 * i + 3);
      }

      VkDeviceSize indexBufferSize = indexBuffer.size() * sizeof(indexBuffer[0]);

      vkWaitForFences(device_, 1, &transferFence_, VK_TRUE, UINT64_MAX);

      std::memcpy(gaussianPly_.staging.ptr, indexBuffer.data(), indexBufferSize);

      VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(transferCommandBuffer_, &beginInfo);

      VkBufferCopy region = {};
      region.srcOffset = 0;
      region.dstOffset = 0;
      region.size = indexBufferSize;
      vkCmdCopyBuffer(transferCommandBuffer_, gaussianPly_.staging.buffer, gaussianIndexBuffer.buffer, 1, &region);

      vkEndCommandBuffer(transferCommandBuffer_);

      VkCommandBufferSubmitInfo commandBufferSubmitInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
      commandBufferSubmitInfo.commandBuffer = transferCommandBuffer_;
      VkSubmitInfo2 submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submitInfo.commandBufferInfoCount = 1;
      submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
      vkResetFences(device_, 1, &transferFence_);
      vkQueueSubmit2(transferQueue_, 1, &submitInfo, transferFence_);
    }

    // TODO: destroy staging buffer once meantime
    vkWaitForFences(device_, 1, &transferFence_, VK_TRUE, UINT64_MAX);
    vmaDestroyBuffer(allocator_, gaussianPly_.staging.buffer, gaussianPly_.staging.allocation);
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
  VkDescriptorSetLayout splatSetLayout_ = VK_NULL_HANDLE;
  VkPipelineLayout graphicsPipelineLayout_ = VK_NULL_HANDLE;

  // descriptors and uniform buffers
  std::vector<VkDescriptorSet> cameraSets_;
  std::vector<Buffer> cameraBuffers_;
  std::vector<VkDescriptorSet> splatSets_;

  // pipelines
  VkPipeline colorTrianglePipeline_ = VK_NULL_HANDLE;
  VkPipeline splatPipeline_ = VK_NULL_HANDLE;

  // swapchain
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  VkPresentModeKHR swapchainPresentMode_ = VK_PRESENT_MODE_FIFO_KHR;
  uint32_t swapchainWidth_ = 0;
  uint32_t swapchainHeight_ = 0;
  std::vector<VkImageView> swapchainImageViews_;

  // framebuffer
  struct Attachment {
    VkImage image;
    VmaAllocation allocation;
    VkImageView view;
  };
  Attachment depthAttachment_ = {};
  std::vector<VkFramebuffer> framebuffers_;

  // transfer
  VkCommandBuffer transferCommandBuffer_ = VK_NULL_HANDLE;
  VkSemaphore transferSemaphore0_ = VK_NULL_HANDLE;
  VkSemaphore transferSemaphore1_ = VK_NULL_HANDLE;
  VkFence transferFence_ = VK_NULL_HANDLE;
  Buffer staging_ = {};

  // gaussian
  VkDescriptorSetLayout gaussianCameraSetLayout_ = VK_NULL_HANDLE;  // uniform
  VkDescriptorSetLayout gaussianSetLayout_ = VK_NULL_HANDLE;        // storage x 4
  VkDescriptorSetLayout gaussianQuadSetLayout_ = VK_NULL_HANDLE;    // storage x 7
  VkDescriptorSetLayout gaussianPlySetLayout_ = VK_NULL_HANDLE;     // storage
  VkPipelineLayout gaussianSplatPipelineLayout_ = VK_NULL_HANDLE;
  VkPipeline parsePlyPipeline_ = VK_NULL_HANDLE;
  VkPipeline rankPipeline_ = VK_NULL_HANDLE;
  VkPipeline inverseIndexPipeline_ = VK_NULL_HANDLE;
  VkPipeline projectionPipeline_ = VK_NULL_HANDLE;
  VrdxSorter sorter_ = VK_NULL_HANDLE;

  std::vector<VkCommandBuffer> gaussianComputeCommandBuffers_;

  struct TimelineSemaphore {
    VkSemaphore semaphore;
    uint64_t timeline;
  };
  std::vector<TimelineSemaphore> gaussianTimelineSemaphores_;

  std::vector<VkFence> gaussianComputeFences_;

  struct GaussianCamera {
    glm::mat4 projection;
    glm::mat4 view;
    glm::vec3 camera_position;
    alignas(16) glm::uvec2 screen_size;  // (width, height)
  };

  struct GaussianSplatPushConstant {
    glm::mat4 model;
    uint32_t pointCount;
  };

  std::vector<Buffer> gaussianCameraBuffers_;

  struct GaussianDescriptorSet {
    VkDescriptorSet camera;
    VkDescriptorSet gaussian;
    VkDescriptorSet quad;
    VkDescriptorSet ply;
  };
  std::vector<GaussianDescriptorSet> gaussianSets_;

  struct GaussianPly {
    Buffer plyBuffer;
    Buffer staging;
  };
  GaussianPly gaussianPly_ = {};
  VkSemaphore gaussianPlyTransferSemaphore_ = VK_NULL_HANDLE;

  struct GaussianSplat {
    glm::mat4 model;
    uint32_t pointCount;
    Buffer position;  // (N, 3)
    Buffer cov3d;     // (N, 6)
    Buffer opacity;   // (N)
    Buffer sh;        // (N, 3, 16) float16
  };
  GaussianSplat gaussianSplat_ = {};

  struct GaussianSplatStorage {
    Buffer visiblePointCount;         // (1)
    Buffer visiblePointCountStaging;  // (1)
    Buffer key;                       // (N). shared with inverse_index.
    Buffer value;                     // (N)
    Buffer storage;                   // (M), by sorter.
    Buffer inverse;                   // (N)
  };
  std::vector<GaussianSplatStorage> gaussianStorages_;

  struct GaussianSplatQuad {
    Buffer indirect;  // (8), VkDrawIndexedIndirectCommand
    Buffer quad;      // (N, 12)
  };
  std::vector<GaussianSplatQuad> gaussianQuads_;
  Buffer gaussianIndexBuffer;  //(96)

  // render
  std::vector<VkCommandBuffer> graphicsCommandBuffers_;
  std::vector<VkSemaphore> imageAcquiredSemaphores_;
  std::vector<VkSemaphore> renderFinishedSemaphores_;
  std::vector<VkFence> renderFinishedFences_;

  struct FrameInfo {
    VkQueryPool computeQueryPool;
    VkQueryPool graphicsQueryPool;

    uint32_t visiblePointCount;

    uint64_t rankTime;
    uint64_t sortTime;
    uint64_t inverseTime;
    uint64_t projectionTime;
    uint64_t drawTime;

    uint64_t totalComputeTime;
    uint64_t totalGraphicsTime;
  };
  std::vector<FrameInfo> frameInfos_;

  // dirty flags
  bool shouldRecreateSwapchain_ = false;
  bool shouldRecreateRenderPass_ = false;
  bool shouldRecreateFramebuffer_ = false;
  bool shouldRecreateGraphicsPipelines_ = false;
  bool shouldRecreateUi_ = false;

  // counters
  int imageAcquireIndex_ = 0;
  int renderIndex_ = 0;
  uint64_t frameCounter_ = 0;
};

Engine::Engine() : impl_(std::make_shared<Impl>()) {}

Engine::~Engine() = default;

void Engine::Run(const std::string& plyFilepath) { impl_->Run(plyFilepath); }

}  // namespace vkgs
