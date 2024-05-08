#ifndef VKGS_ENGINE_VULKAN_PIPELINE_LAYOUT_H
#define VKGS_ENGINE_VULKAN_PIPELINE_LAYOUT_H

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/descriptor_layout.h"

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

#endif  // VKGS_ENGINE_VULKAN_PIPELINE_LAYOUT_H
