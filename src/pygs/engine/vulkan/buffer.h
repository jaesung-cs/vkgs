#ifndef PYGS_ENGINE_VULKAN_BUFFER_H
#define PYGS_ENGINE_VULKAN_BUFFER_H

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "context.h"

namespace pygs {
namespace vk {

class Buffer {
 public:
  Buffer();

  Buffer(Context context, VkDeviceSize size, VkBufferUsageFlags usage);

  ~Buffer();

  operator VkBuffer() const;

  void FromCpu(VkCommandBuffer command_buffer, const void* src,
               VkDeviceSize size);

  template <typename T>
  void FromCpu(VkCommandBuffer command_buffer, const std::vector<T>& v) {
    FromCpu(command_buffer, v.data(), v.size() * sizeof(T));
  }

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_BUFFER_H
