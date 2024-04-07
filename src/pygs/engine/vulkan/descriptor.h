#ifndef PYGS_ENGINE_VULKAN_DESCRIPTOR_H
#define PYGS_ENGINE_VULKAN_DESCRIPTOR_H

#include <memory>

#include <vulkan/vulkan.h>

#include "pygs/engine/vulkan/context.h"
#include "pygs/engine/vulkan/descriptor_layout.h"

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
  void UpdateInputAttachment(uint32_t binding, VkImageView image_view);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_DESCRIPTOR_H
