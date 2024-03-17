#ifndef PYGS_ENGINE_VULKAN_CONTEXT_H
#define PYGS_ENGINE_VULKAN_CONTEXT_H

#include <memory>
#include <vector>
#include <string>

#include <vulkan/vulkan.h>

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
  VmaAllocator allocator() const noexcept;

  VkResult GetMemoryFdKHR(const VkMemoryGetFdInfoKHR* pGetFdInfo, int* pFd);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_CONTEXT_H
