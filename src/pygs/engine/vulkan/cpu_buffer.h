#ifndef PYGS_ENGINE_VULKAN_CPU_BUFFER_H
#define PYGS_ENGINE_VULKAN_CPU_BUFFER_H

#include <memory>

#include <vulkan/vulkan.h>

#include "context.h"

namespace pygs {
namespace vk {

class CpuBuffer {
 public:
  CpuBuffer();

  CpuBuffer(Context context, VkDeviceSize size);

  ~CpuBuffer();

  operator VkBuffer() const;

  const void* data() const;
  VkDeviceSize size() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_CPU_BUFFER_H
