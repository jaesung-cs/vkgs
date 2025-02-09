#ifndef VKGS_VULKAN_BUFFER_H
#define VKGS_VULKAN_BUFFER_H

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "vkgs/vulkan/context.h"

namespace vkgs {
namespace vk {

class Buffer {
 public:
  Buffer();

  Buffer(Context context, VkDeviceSize size, VkBufferUsageFlags usage);

  ~Buffer();

  operator bool() const noexcept { return impl_ != nullptr; }

  operator VkBuffer() const;

  VkDeviceSize size() const;

  void FromCpu(VkCommandBuffer command_buffer, const void* src, VkDeviceSize size);

  template <typename T>
  void FromCpu(VkCommandBuffer command_buffer, const std::vector<T>& v) {
    FromCpu(command_buffer, v.data(), v.size() * sizeof(T));
  }

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_BUFFER_H
