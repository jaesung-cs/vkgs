#ifndef VKGS_ENGINE_VULKAN_COMPUTE_PIPELINE_H
#define VKGS_ENGINE_VULKAN_COMPUTE_PIPELINE_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/pipeline_layout.h"
#include "vkgs/engine/vulkan/shader_module.h"

namespace vkgs {
namespace vk {

struct ComputePipelineCreateInfo {
  PipelineLayout layout;
  ShaderSource source;
};

class ComputePipeline {
 public:
  ComputePipeline();

  ComputePipeline(Context context,
                  const ComputePipelineCreateInfo& create_info);

  ~ComputePipeline();

  operator VkPipeline() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_COMPUTE_PIPELINE_H
