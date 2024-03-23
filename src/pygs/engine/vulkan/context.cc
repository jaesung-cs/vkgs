#include "context.h"

#include <iostream>
#include <vector>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace pygs {
namespace vk {
namespace {

static VKAPI_ATTR VkBool32 VKAPI_CALL
DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
              void* pUserData) {
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl
            << std::endl;

  return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

}  // namespace

class Context::Impl {
 public:
  Impl() {
    // Instance
    VkApplicationInfo application_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application_info.pApplicationName = "rtgs";
    application_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 0, 0);
    application_info.pEngineName = "rtgs_vulkan";
    application_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 0, 0);
    application_info.apiVersion = VK_API_VERSION_1_3;

    VkDebugUtilsMessengerCreateInfoEXT messenger_info = {
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messenger_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messenger_info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messenger_info.pfnUserCallback = DebugCallback;

    std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};

    uint32_t count;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char*> instance_extensions(glfw_extensions,
                                                 glfw_extensions + count);
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkInstanceCreateInfo instance_info = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pNext = &messenger_info;
    instance_info.pApplicationInfo = &application_info;
    instance_info.enabledLayerCount = layers.size();
    instance_info.ppEnabledLayerNames = layers.data();
    instance_info.enabledExtensionCount = instance_extensions.size();
    instance_info.ppEnabledExtensionNames = instance_extensions.data();
    vkCreateInstance(&instance_info, NULL, &instance_);

    CreateDebugUtilsMessengerEXT(instance_, &messenger_info, NULL, &messenger_);

    // Physical device
    uint32_t physical_device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &physical_device_count, NULL);
    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    vkEnumeratePhysicalDevices(instance_, &physical_device_count,
                               physical_devices.data());
    physical_device_ = physical_devices[0];

    // Find graphics queue
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_,
                                             &queue_family_count, NULL);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device_, &queue_family_count, queue_families.data());

    constexpr VkQueueFlags graphics_queue_flags = VK_QUEUE_GRAPHICS_BIT;
    for (int i = 0; i < queue_families.size(); i++) {
      const auto& queue_family = queue_families[i];
      bool proper_queue_type = (queue_family.queueFlags &
                                graphics_queue_flags) == graphics_queue_flags;
      bool presentation_support = glfwGetPhysicalDevicePresentationSupport(
          instance_, physical_device_, i);
      if (proper_queue_type && presentation_support) {
        queue_family_index_ = i;
        break;
      }
    }

    // features
    VkPhysicalDeviceSynchronization2Features synchronization_2_features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};

    VkPhysicalDeviceImagelessFramebufferFeatures imageless_framebuffer_feature =
        {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES};
    imageless_framebuffer_feature.pNext = &synchronization_2_features;

    VkPhysicalDeviceFeatures2 features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &imageless_framebuffer_feature;
    vkGetPhysicalDeviceFeatures2(physical_device_, &features);

    // queues
    std::vector<float> queue_priorities = {
        1.f,
    };
    std::vector<VkDeviceQueueCreateInfo> queue_infos(1);
    queue_infos[0] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_infos[0].queueFamilyIndex = queue_family_index_;
    queue_infos[0].queueCount = queue_priorities.size();
    queue_infos[0].pQueuePriorities = queue_priorities.data();

    std::vector<const char*> device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_IMAGELESS_FRAMEBUFFER_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
#ifdef _WIN32
        "VK_KHR_external_memory_win32",
        "VK_KHR_external_semaphore_win32",
#else
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
#endif
    };

    VkDeviceCreateInfo device_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &features;
    device_info.queueCreateInfoCount = queue_infos.size();
    device_info.pQueueCreateInfos = queue_infos.data();
    device_info.enabledExtensionCount = device_extensions.size();
    device_info.ppEnabledExtensionNames = device_extensions.data();
    vkCreateDevice(physical_device_, &device_info, NULL, &device_);

    vkGetDeviceQueue(device_, queue_family_index_, 0, &queue_);

    // Extensions
    GetMemoryFdKHR_ =
        (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR");
    GetSemaphoreFdKHR_ = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
        device_, "vkGetSemaphoreFdKHR");

#ifdef _WIN32
    GetMemoryWin32HandleKHR_ =
        (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(
            device_, "vkGetMemoryWin32HandleKHR");
    GetSemaphoreWin32HandleKHR_ =
        (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(
            device_, "vkGetSemaphoreWin32HandleKHR");
#endif

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice = physical_device_;
    allocator_info.device = device_;
    allocator_info.instance = instance_;
    allocator_info.vulkanApiVersion = VK_API_VERSION_1_3;
    vmaCreateAllocator(&allocator_info, &allocator_);

    VkCommandPoolCreateInfo command_pool_info = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                              VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = queue_family_index_;
    vkCreateCommandPool(device_, &command_pool_info, NULL, &command_pool_);

    std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1048576},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2048},
    };
    VkDescriptorPoolCreateInfo descriptor_pool_info = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptor_pool_info.maxSets = 1048576;
    descriptor_pool_info.poolSizeCount = pool_sizes.size();
    descriptor_pool_info.pPoolSizes = pool_sizes.data();
    vkCreateDescriptorPool(device_, &descriptor_pool_info, NULL,
                           &descriptor_pool_);
  }

  ~Impl() {
    vkDestroyDescriptorPool(device_, descriptor_pool_, NULL);
    vkDestroyCommandPool(device_, command_pool_, NULL);
    vmaDestroyAllocator(allocator_);
    vkDestroyDevice(device_, NULL);

    DestroyDebugUtilsMessengerEXT(instance_, messenger_, NULL);
    vkDestroyInstance(instance_, NULL);
  }

  VkInstance instance() const noexcept { return instance_; }
  VkPhysicalDevice physical_device() const noexcept { return physical_device_; }
  VkDevice device() const noexcept { return device_; }
  VkQueue queue() const noexcept { return queue_; }
  VmaAllocator allocator() const noexcept { return allocator_; }
  VkCommandPool command_pool() const noexcept { return command_pool_; }
  VkDescriptorPool descriptor_pool() const noexcept { return descriptor_pool_; }

  VkResult GetMemoryFdKHR(const VkMemoryGetFdInfoKHR* pGetFdInfo, int* pFd) {
    if (GetMemoryFdKHR_ == nullptr) return VK_ERROR_EXTENSION_NOT_PRESENT;
    return GetMemoryFdKHR_(device_, pGetFdInfo, pFd);
  }

  VkResult GetSemaphoreFdKHR(const VkSemaphoreGetFdInfoKHR* pGetFdInfo,
                             int* pFd) {
    if (GetSemaphoreFdKHR_ == nullptr) return VK_ERROR_EXTENSION_NOT_PRESENT;
    return GetSemaphoreFdKHR_(device_, pGetFdInfo, pFd);
  }

#ifdef _WIN32
  VkResult GetMemoryWin32HandleKHR(
      const VkMemoryGetWin32HandleInfoKHR* pGetWin32HandleInfo,
      HANDLE* handle) {
    if (GetMemoryWin32HandleKHR_ == nullptr)
      return VK_ERROR_EXTENSION_NOT_PRESENT;
    return GetMemoryWin32HandleKHR_(device_, pGetWin32HandleInfo, handle);
  }

  VkResult GetSemaphoreWin32HandleKHR(
      const VkSemaphoreGetWin32HandleInfoKHR* pGetWin32HandleInfo,
      HANDLE* pFd) {
    if (GetSemaphoreWin32HandleKHR_ == nullptr)
      return VK_ERROR_EXTENSION_NOT_PRESENT;
    return GetSemaphoreWin32HandleKHR_(device_, pGetWin32HandleInfo, pFd);
  }
#endif

 private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  uint32_t queue_family_index_ = 0;
  VkQueue queue_ = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

  PFN_vkGetMemoryFdKHR GetMemoryFdKHR_ = nullptr;
  PFN_vkGetSemaphoreFdKHR GetSemaphoreFdKHR_ = nullptr;

#ifdef _WIN32
  PFN_vkGetMemoryWin32HandleKHR GetMemoryWin32HandleKHR_ = nullptr;
  PFN_vkGetSemaphoreWin32HandleKHR GetSemaphoreWin32HandleKHR_ = nullptr;
#endif
};

Context::Context() = default;

Context::Context(int) : impl_(std::make_shared<Impl>()) {}

Context::~Context() = default;

VkInstance Context::instance() const noexcept { return impl_->instance(); }

VkPhysicalDevice Context::physical_device() const noexcept {
  return impl_->physical_device();
}

VkDevice Context::device() const noexcept { return impl_->device(); }

VkQueue Context::queue() const noexcept { return impl_->queue(); }

VmaAllocator Context::allocator() const noexcept { return impl_->allocator(); }

VkCommandPool Context::command_pool() const noexcept {
  return impl_->command_pool();
}

VkDescriptorPool Context::descriptor_pool() const noexcept {
  return impl_->descriptor_pool();
}

VkResult Context::GetMemoryFdKHR(const VkMemoryGetFdInfoKHR* pGetFdInfo,
                                 int* pFd) {
  return impl_->GetMemoryFdKHR(pGetFdInfo, pFd);
}

VkResult Context::GetSemaphoreFdKHR(const VkSemaphoreGetFdInfoKHR* pGetFdInfo,
                                    int* pFd) {
  return impl_->GetSemaphoreFdKHR(pGetFdInfo, pFd);
}

#ifdef _WIN32
VkResult Context::GetMemoryWin32HandleKHR(
    const VkMemoryGetWin32HandleInfoKHR* pGetFdInfo, HANDLE* pHandle) {
  return impl_->GetMemoryWin32HandleKHR(pGetFdInfo, pHandle);
}

VkResult Context::GetSemaphoreWin32HandleKHR(
    const VkSemaphoreGetWin32HandleInfoKHR* pGetFdInfo, HANDLE* pHandle) {
  return impl_->GetSemaphoreWin32HandleKHR(pGetFdInfo, pHandle);
}
#endif

}  // namespace vk
}  // namespace pygs
