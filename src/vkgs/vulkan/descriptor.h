#ifndef VKGS_VULKAN_DESCRIPTOR_H
#define VKGS_VULKAN_DESCRIPTOR_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/vulkan/context.h"
#include "vkgs/vulkan/descriptor_layout.h"

namespace vkgs {
namespace vk {

class Descriptor {
 public:
  Descriptor();

  Descriptor(Context context, DescriptorLayout layout);

  ~Descriptor();

  operator VkDescriptorSet() const;

  void Update(uint32_t binding, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size = 0);
  void UpdateInputAttachment(uint32_t binding, VkImageView image_view);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_DESCRIPTOR_H
