#ifndef VKGS_VULKAN_DESCRIPTOR_LAYOUT_H
#define VKGS_VULKAN_DESCRIPTOR_LAYOUT_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/vulkan/context.h"

namespace vkgs {
namespace vk {

struct DescriptorLayoutBinding {
  uint32_t binding = 0;
  VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_SAMPLER;
  VkShaderStageFlags stage_flags = 0;
};

struct DescriptorLayoutCreateInfo {
  std::vector<DescriptorLayoutBinding> bindings;
};

class DescriptorLayout {
 public:
  DescriptorLayout();
  DescriptorLayout(Context context,
                   const DescriptorLayoutCreateInfo& create_info);
  ~DescriptorLayout();

  operator VkDescriptorSetLayout() const;

  VkDescriptorType type(uint32_t binding) const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_DESCRIPTOR_LAYOUT_H
