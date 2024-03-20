#ifndef PYGS_ENGINE_VULKAN_DESCRIPTOR_H
#define PYGS_ENGINE_VULKAN_DESCRIPTOR_H

#include <memory>

#include <vulkan/vulkan.h>

#include "context.h"
#include "descriptor_layout.h"

namespace pygs {
namespace vk {

class Descriptor {
 public:
  Descriptor();

  Descriptor(Context context, DescriptorLayout layout);

  ~Descriptor();

  operator VkDescriptorSet() const;

  void Update(uint32_t binding, VkBuffer buffer, VkDeviceSize offset,
              VkDeviceSize size);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_DESCRIPTOR_H
