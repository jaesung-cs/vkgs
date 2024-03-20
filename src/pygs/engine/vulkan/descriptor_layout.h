#ifndef PYGS_ENGINE_VULKAN_DESCRIPTOR_LAYOUT_H
#define PYGS_ENGINE_VULKAN_DESCRIPTOR_LAYOUT_H

#include <memory>

#include <vulkan/vulkan.h>

#include "context.h"

namespace pygs {
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
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_DESCRIPTOR_LAYOUT_H
