#ifndef VKGS_ENGINE_VULKAN_DESCRIPTOR_H
#define VKGS_ENGINE_VULKAN_DESCRIPTOR_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/descriptor_layout.h"

namespace vkgs {
namespace vk {

class Descriptor {
 public:
  Descriptor();

  Descriptor(Context context, DescriptorLayout layout);

  ~Descriptor();

  operator VkDescriptorSet() const;

  void Update(uint32_t binding, VkBuffer buffer, VkDeviceSize offset,
              VkDeviceSize size = 0);
  void UpdateInputAttachment(uint32_t binding, VkImageView image_view);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_DESCRIPTOR_H
