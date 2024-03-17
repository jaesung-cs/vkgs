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
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

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
    VkPhysicalDeviceImagelessFramebufferFeatures
        imageless_framebuffer_features = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES};

    VkPhysicalDeviceFeatures2 features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &imageless_framebuffer_features;
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
    };

    VkDeviceCreateInfo device_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &features;
    device_info.queueCreateInfoCount = queue_infos.size();
    device_info.pQueueCreateInfos = queue_infos.data();
    device_info.enabledExtensionCount = device_extensions.size();
    device_info.ppEnabledExtensionNames = device_extensions.data();
    vkCreateDevice(physical_device_, &device_info, NULL, &device_);

    vkGetDeviceQueue(device_, queue_family_index_, 0, &queue_);

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice = physical_device_;
    allocator_info.device = device_;
    allocator_info.instance = instance_;
    allocator_info.vulkanApiVersion = VK_API_VERSION_1_3;
    vmaCreateAllocator(&allocator_info, &allocator_);
  }

  ~Impl() {
    vmaDestroyAllocator(allocator_);
    vkDestroyDevice(device_, NULL);

    DestroyDebugUtilsMessengerEXT(instance_, messenger_, NULL);
    vkDestroyInstance(instance_, NULL);
  }

  VkInstance instance() const noexcept { return instance_; }
  VkPhysicalDevice physical_device() const noexcept { return physical_device_; }
  VkDevice device() const noexcept { return device_; }
  VmaAllocator allocator() const noexcept { return allocator_; }

 private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  uint32_t queue_family_index_ = 0;
  VkQueue queue_ = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
};

Context::Context() : impl_(std::make_shared<Impl>()) {}

Context::~Context() = default;

VkInstance Context::instance() const noexcept { return impl_->instance(); }

VkPhysicalDevice Context::physical_device() const noexcept {
  return impl_->physical_device();
}

VkDevice Context::device() const noexcept { return impl_->device(); }

VmaAllocator Context::allocator() const noexcept { return impl_->allocator(); }

}  // namespace vk
}  // namespace pygs
