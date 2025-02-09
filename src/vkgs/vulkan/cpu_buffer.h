#ifndef VKGS_VULKAN_CPU_BUFFER_H
#define VKGS_VULKAN_CPU_BUFFER_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/vulkan/context.h"

namespace vkgs {
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
}  // namespace vkgs

#endif  // VKGS_VULKAN_CPU_BUFFER_H
