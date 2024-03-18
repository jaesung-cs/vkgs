#ifndef PYGS_ENGINE_VULKAN_CONTEXT_H
#define PYGS_ENGINE_VULKAN_CONTEXT_H

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

namespace pygs {
namespace vk {

class Context {
 public:
  Context();

  ~Context();

  VkInstance instance() const noexcept;
  VkPhysicalDevice physical_device() const noexcept;
  VkDevice device() const noexcept;
  VkQueue queue() const noexcept;
  VmaAllocator allocator() const noexcept;
  VkCommandPool command_pool() const noexcept;

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
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_CONTEXT_H
