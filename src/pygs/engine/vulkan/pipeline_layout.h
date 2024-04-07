#ifndef PYGS_ENGINE_VULKAN_PIPELINE_LAYOUT_H
#define PYGS_ENGINE_VULKAN_PIPELINE_LAYOUT_H

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "pygs/engine/vulkan/context.h"
#include "pygs/engine/vulkan/descriptor_layout.h"

namespace pygs {
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
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_PIPELINE_LAYOUT_H
