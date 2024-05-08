#include "context.h"

#include <iostream>
#include <fstream>
#include <vector>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace vkgs {
namespace vk {
namespace {

const std::string pipeline_cache_filename = "pipeline_cache.bin";

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
    // instance
    VkApplicationInfo application_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application_info.pApplicationName = "rtgs";
    application_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 0, 0);
    application_info.pEngineName = "rtgs_vulkan";
    application_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 0, 0);
    application_info.apiVersion = VK_API_VERSION_1_2;

    VkDebugUtilsMessengerCreateInfoEXT messenger_info = {
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messenger_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messenger_info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messenger_info.pfnUserCallback = DebugCallback;

    std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};

    uint32_t count;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char*> instance_extensions(glfw_extensions,
                                                 glfw_extensions + count);
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instance_extensions.push_back(
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instance_extensions.push_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    VkInstanceCreateInfo instance_info = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pNext = &messenger_info;
    instance_info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    instance_info.pApplicationInfo = &application_info;
    instance_info.enabledLayerCount = layers.size();
    instance_info.ppEnabledLayerNames = layers.data();
    instance_info.enabledExtensionCount = instance_extensions.size();
    instance_info.ppEnabledExtensionNames = instance_extensions.data();
    vkCreateInstance(&instance_info, NULL, &instance_);

    CreateDebugUtilsMessengerEXT(instance_, &messenger_info, NULL, &messenger_);

    // physical device
    uint32_t physical_device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &physical_device_count, NULL);
    if (physical_device_count == 0) throw std::runtime_error("No GPU found");

    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    vkEnumeratePhysicalDevices(instance_, &physical_device_count,
                               physical_devices.data());

    uint32_t physical_device_id = 0;
    uint32_t physical_device_score = 0;
    for (int i = 0; i < physical_device_count; ++i) {
      uint32_t score = 0;
      VkPhysicalDevice physical_device = physical_devices[i];

      // +1 for discrete GPU
      VkPhysicalDeviceProperties physical_device_properties;
      vkGetPhysicalDeviceProperties(physical_device,
                                    &physical_device_properties);
      if (physical_device_properties.deviceType ==
          VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 1;
      }

      // +10 for having graphics queue with presention support
      uint32_t queue_family_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                               &queue_family_count, NULL);
      std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
      vkGetPhysicalDeviceQueueFamilyProperties(
          physical_device, &queue_family_count, queue_families.data());

      constexpr VkQueueFlags graphics_queue_flags = VK_QUEUE_GRAPHICS_BIT;
      for (int i = 0; i < queue_families.size(); ++i) {
        const auto& queue_family = queue_families[i];

        bool is_graphics_queue_type =
            (queue_family.queueFlags & graphics_queue_flags) ==
            graphics_queue_flags;
        bool presentation_support = glfwGetPhysicalDevicePresentationSupport(
            instance_, physical_device, i);

        if (is_graphics_queue_type && presentation_support) {
          score += 10;
          break;
        }
      }

      if (physical_device_score < score) {
        physical_device_id = i;
        physical_device_score = score;
      }
    }

    physical_device_ = physical_devices[physical_device_id];

    // physical device properties
    VkPhysicalDeviceProperties physical_device_properties;
    vkGetPhysicalDeviceProperties(physical_device_,
                                  &physical_device_properties);
    device_name_ = physical_device_properties.deviceName;

    std::cout << device_name_ << std::endl;

    // find graphics queue
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_,
                                             &queue_family_count, NULL);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device_, &queue_family_count, queue_families.data());

    constexpr VkQueueFlags graphics_queue_flags = VK_QUEUE_GRAPHICS_BIT;
    constexpr VkQueueFlags transfer_queue_flags =
        VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT;
    for (int i = 0; i < queue_families.size(); ++i) {
      const auto& queue_family = queue_families[i];

      bool is_graphics_queue_type =
          (queue_family.queueFlags & graphics_queue_flags) ==
          graphics_queue_flags;
      bool presentation_support = glfwGetPhysicalDevicePresentationSupport(
          instance_, physical_device_, i);

      bool is_transfer_queue_type =
          queue_family.queueFlags == transfer_queue_flags;

      if (is_graphics_queue_type && presentation_support) {
        graphics_queue_family_index_ = i;
      }
      if (!is_graphics_queue_type && is_transfer_queue_type) {
        transfer_queue_family_index_ = i;
      }
    }

    // features
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore_features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};

    VkPhysicalDeviceImagelessFramebufferFeatures imageless_framebuffer_feature =
        {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES};
    imageless_framebuffer_feature.pNext = &timeline_semaphore_features;

    VkPhysicalDeviceFeatures2 features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &imageless_framebuffer_feature;
    vkGetPhysicalDeviceFeatures2(physical_device_, &features);

    // queues
    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    std::vector<float> queue_priorities = {0.5f, 1.f};

    if (graphics_queue_family_index_ != transfer_queue_family_index_) {
      queue_infos.resize(2);
      queue_infos[0] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
      queue_infos[0].queueFamilyIndex = graphics_queue_family_index_;
      queue_infos[0].queueCount = 1;
      queue_infos[0].pQueuePriorities = &queue_priorities[0];

      queue_infos[1] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
      queue_infos[1].queueFamilyIndex = transfer_queue_family_index_;
      queue_infos[1].queueCount = 1;
      queue_infos[1].pQueuePriorities = &queue_priorities[1];
    } else {
      queue_infos.resize(1);
      queue_infos[0] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
      queue_infos[0].queueFamilyIndex = graphics_queue_family_index_;
      queue_infos[0].queueCount = 2;
      queue_infos[0].pQueuePriorities = &queue_priorities[0];
    }

    std::vector<const char*> device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#ifdef _WIN32
        "VK_KHR_external_memory_win32",
        "VK_KHR_external_semaphore_win32",
#elif __APPLE__
        "VK_KHR_portability_subset",
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

    vkGetDeviceQueue(device_, graphics_queue_family_index_, 0,
                     &graphics_queue_);

    if (transfer_queue_family_index_ == graphics_queue_family_index_) {
      vkGetDeviceQueue(device_, transfer_queue_family_index_, 1,
                       &transfer_queue_);
    } else {
      vkGetDeviceQueue(device_, transfer_queue_family_index_, 0,
                       &transfer_queue_);
    }

    vkGetDeviceQueue(device_, graphics_queue_family_index_, 0,
                     &graphics_queue_);
    vkGetDeviceQueue(device_, transfer_queue_family_index_, 0,
                     &transfer_queue_);

    // extensions
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
    allocator_info.vulkanApiVersion = application_info.apiVersion;
    vmaCreateAllocator(&allocator_info, &allocator_);

    VkCommandPoolCreateInfo command_pool_info = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                              VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = graphics_queue_family_index_;
    vkCreateCommandPool(device_, &command_pool_info, NULL, &command_pool_);

    std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2048},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 64},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 64},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
    };
    VkDescriptorPoolCreateInfo descriptor_pool_info = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptor_pool_info.flags =
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptor_pool_info.maxSets = 2048;
    descriptor_pool_info.poolSizeCount = pool_sizes.size();
    descriptor_pool_info.pPoolSizes = pool_sizes.data();
    vkCreateDescriptorPool(device_, &descriptor_pool_info, NULL,
                           &descriptor_pool_);

    std::vector<char> pipeline_cache_data;
    {
      std::ifstream in(pipeline_cache_filename, std::ios::binary);
      if (in.is_open()) {
        pipeline_cache_data =
            std::vector<char>(std::istreambuf_iterator<char>(in), {});
      }
    }

    VkPipelineCacheCreateInfo pipeline_cache_info = {
        VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    pipeline_cache_info.initialDataSize = pipeline_cache_data.size();
    pipeline_cache_info.pInitialData = pipeline_cache_data.data();
    vkCreatePipelineCache(device_, &pipeline_cache_info, NULL,
                          &pipeline_cache_);
  }

  ~Impl() {
    size_t size;
    vkGetPipelineCacheData(device_, pipeline_cache_, &size, NULL);
    if (size > 0) {
      std::vector<char> data(size);
      vkGetPipelineCacheData(device_, pipeline_cache_, &size, data.data());

      std::ofstream out(pipeline_cache_filename, std::ios::binary);
      if (out.is_open()) {
        out.write(data.data(), data.size());
      }
    }

    vkDestroyPipelineCache(device_, pipeline_cache_, NULL);
    vkDestroyDescriptorPool(device_, descriptor_pool_, NULL);
    vkDestroyCommandPool(device_, command_pool_, NULL);
    vmaDestroyAllocator(allocator_);
    vkDestroyDevice(device_, NULL);

    DestroyDebugUtilsMessengerEXT(instance_, messenger_, NULL);
    vkDestroyInstance(instance_, NULL);
  }

  const std::string& device_name() const noexcept { return device_name_; }
  VkInstance instance() const noexcept { return instance_; }
  VkPhysicalDevice physical_device() const noexcept { return physical_device_; }
  VkDevice device() const noexcept { return device_; }
  uint32_t graphics_queue_family_index() const noexcept {
    return graphics_queue_family_index_;
  }
  uint32_t transfer_queue_family_index() const noexcept {
    return transfer_queue_family_index_;
  }
  VkQueue graphics_queue() const noexcept { return graphics_queue_; }
  VkQueue transfer_queue() const noexcept { return transfer_queue_; }
  VmaAllocator allocator() const noexcept { return allocator_; }
  VkCommandPool command_pool() const noexcept { return command_pool_; }
  VkDescriptorPool descriptor_pool() const noexcept { return descriptor_pool_; }
  VkPipelineCache pipeline_cache() const noexcept { return pipeline_cache_; }

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
  uint32_t graphics_queue_family_index_ = 0;
  uint32_t transfer_queue_family_index_ = 0;
  VkQueue graphics_queue_ = VK_NULL_HANDLE;
  VkQueue transfer_queue_ = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;

  std::string device_name_;

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

const std::string& Context::device_name() const { return impl_->device_name(); }
VkInstance Context::instance() const { return impl_->instance(); }

VkPhysicalDevice Context::physical_device() const {
  return impl_->physical_device();
}

VkDevice Context::device() const { return impl_->device(); }

uint32_t Context::graphics_queue_family_index() const {
  return impl_->graphics_queue_family_index();
}

uint32_t Context::transfer_queue_family_index() const {
  return impl_->transfer_queue_family_index();
}

VkQueue Context::graphics_queue() const { return impl_->graphics_queue(); }

VkQueue Context::transfer_queue() const { return impl_->transfer_queue(); }

VmaAllocator Context::allocator() const { return impl_->allocator(); }

VkCommandPool Context::command_pool() const { return impl_->command_pool(); }

VkDescriptorPool Context::descriptor_pool() const {
  return impl_->descriptor_pool();
}

VkPipelineCache Context::pipeline_cache() const {
  return impl_->pipeline_cache();
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
}  // namespace vkgs
