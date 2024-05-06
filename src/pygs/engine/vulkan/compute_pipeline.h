#ifndef PYGS_ENGINE_VULKAN_COMPUTE_PIPELINE_H
#define PYGS_ENGINE_VULKAN_COMPUTE_PIPELINE_H

#include <memory>

#include <vulkan/vulkan.h>

#include "pygs/engine/vulkan/context.h"
#include "pygs/engine/vulkan/pipeline_layout.h"
#include "pygs/engine/vulkan/shader_module.h"

namespace pygs {
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
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_COMPUTE_PIPELINE_H
