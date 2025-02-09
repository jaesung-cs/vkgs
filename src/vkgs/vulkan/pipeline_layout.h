#ifndef VKGS_VULKAN_PIPELINE_LAYOUT_H
#define VKGS_VULKAN_PIPELINE_LAYOUT_H

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "vkgs/vulkan/context.h"
#include "vkgs/vulkan/descriptor_layout.h"

namespace vkgs {
namespace vk {

struct PipelineLayoutCreateInfo {
  std::vector<DescriptorLayout> layouts;
  std::vector<VkPushConstantRange> push_constants;
};

class PipelineLayout {
 public:
  PipelineLayout();

  PipelineLayout(Context context, const PipelineLayoutCreateInfo& create_info);

  ~PipelineLayout();

  operator VkPipelineLayout() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_PIPELINE_LAYOUT_H
