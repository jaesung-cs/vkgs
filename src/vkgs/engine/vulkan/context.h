#ifndef VKGS_ENGINE_VULKAN_CONTEXT_H
#define VKGS_ENGINE_VULKAN_CONTEXT_H

#include <memory>
#include <vector>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#endif

#include "vk_mem_alloc.h"

namespace vkgs {
namespace vk {

class Context {
 public:
  Context();

  explicit Context(int);

  ~Context();

  const std::string& device_name() const;
  VkInstance instance() const;
  VkPhysicalDevice physical_device() const;
  VkDevice device() const;
  uint32_t graphics_queue_family_index() const;
  uint32_t transfer_queue_family_index() const;
  VkQueue graphics_queue() const;
  VkQueue transfer_queue() const;
  VmaAllocator allocator() const;
  VkCommandPool command_pool() const;
  VkDescriptorPool descriptor_pool() const;
  VkPipelineCache pipeline_cache() const;

  bool geometry_shader_available() const;

  VkResult GetMemoryFdKHR(const VkMemoryGetFdInfoKHR* pGetFdInfo, int* pFd);
  VkResult GetSemaphoreFdKHR(const VkSemaphoreGetFdInfoKHR* pGetFdInfo,
                             int* pFd);

#ifdef _WIN32
  VkResult Context::GetMemoryWin32HandleKHR(
      const VkMemoryGetWin32HandleInfoKHR* pGetFdInfo, HANDLE* handle);
  VkResult Context::GetSemaphoreWin32HandleKHR(
      const VkSemaphoreGetWin32HandleInfoKHR* pGetFdInfo, HANDLE* handle);
#endif

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_CONTEXT_H
