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

#include "vkgs/viewer/viewer.h"

#include <vkgs/scene/camera.h>
#include <vkgs/util/clock.h>

namespace vkgs {
namespace {

struct Resolution {
  int width;
  int height;
  const char* tag;
};

std::vector<Resolution> preset_resolutions = {
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

}  // namespace

class Engine::Impl {
 private:
  enum class DisplayMode {
    Windowed,
    WindowedFullscreen,
  };

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

    constexpr VkQueueFlags graphicsQueueFlags = VK_QUEUE_GRAPHICS_BIT;
    constexpr VkQueueFlags transferQueueFlags = VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT;
    constexpr VkQueueFlags computeQueueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    for (int i = 0; i < queueFamilies.size(); ++i) {
      const auto& queueFamily = queueFamilies[i];

      bool isGraphicsQueueType = (queueFamily.queueFlags & graphicsQueueFlags) == graphicsQueueFlags;
      bool presentationSupport = glfwGetPhysicalDevicePresentationSupport(instance_, physicalDevice_, i);

      // TODO: proper transfer queue. no optical flow bit.
      bool isTransferQueueType = (queueFamily.queueFlags & transferQueueFlags) == transferQueueFlags &&
                                 !(queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT);

      bool isComputeQueueType = (queueFamily.queueFlags & computeQueueFlags) == computeQueueFlags;

      if (isGraphicsQueueType && presentationSupport) graphicsQueueFamily_ = i;
      if (!isGraphicsQueueType && isTransferQueueType) transferQueueFamily_ = i;
      if (!isGraphicsQueueType && isComputeQueueType) computeQueueFamily_ = i;
    }

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

    for (auto semaphore : imageAcquiredSemaphores_) vkDestroySemaphore(device_, semaphore, NULL);
    for (auto semaphore : renderFinishedSemaphores_) vkDestroySemaphore(device_, semaphore, NULL);
    for (auto fence : renderFinishedFences_) vkDestroyFence(device_, fence, NULL);

    for (auto framebuffer : framebuffers_) vkDestroyFramebuffer(device_, framebuffer, NULL);

    vkDestroyRenderPass(device_, renderPass_, NULL);

    vkDestroyCommandPool(device_, transferCommandPool_, NULL);
    vkDestroyCommandPool(device_, computeCommandPool_, NULL);
    vkDestroyCommandPool(device_, graphicsCommandPool_, NULL);
    vkDestroyDescriptorPool(device_, descriptorPool_, NULL);

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
      ImGui::Text("device: %s", deviceName_.c_str());
      ImGui::Text("fps   : %.2f", io.Framerate);
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
    swapchainInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
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

  void ToggleDisplayMode() {
    switch (displayMode_) {
      case DisplayMode::Windowed:
        SetWindowedFullscreen();
        break;
      case DisplayMode::WindowedFullscreen:
        SetWindowed();
        break;
    }
  }

  void SetWindowed() {
    if (displayMode_ == DisplayMode::WindowedFullscreen) {
      displayMode_ = DisplayMode::Windowed;
      viewer_.SetWindowed();
    }
  }

  void SetWindowedFullscreen() {
    if (displayMode_ == DisplayMode::Windowed) {
      displayMode_ = DisplayMode::WindowedFullscreen;
      viewer_.SetWindowedFullscreen();
    }
  }

  DisplayMode displayMode_ = DisplayMode::Windowed;

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

  VkRenderPass renderPass_ = VK_NULL_HANDLE;

  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  uint32_t swapchainWidth_ = 0;
  uint32_t swapchainHeight_ = 0;
  std::vector<VkImageView> swapchainImageViews_;

  std::vector<VkFramebuffer> framebuffers_;

  std::vector<VkCommandBuffer> graphicsCommandBuffers_;
  std::vector<VkSemaphore> imageAcquiredSemaphores_;
  std::vector<VkSemaphore> renderFinishedSemaphores_;
  std::vector<VkFence> renderFinishedFences_;

  // dirty flags
  bool shouldRecreateSwapchain_ = false;
  bool shouldRecreateRenderPass_ = false;
  bool shouldRecreateFramebuffer_ = false;
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
