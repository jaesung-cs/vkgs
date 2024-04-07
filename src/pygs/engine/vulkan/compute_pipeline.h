#ifndef PYGS_ENGINE_VULKAN_COMPUTE_PIPELINE_H
#define PYGS_ENGINE_VULKAN_COMPUTE_PIPELINE_H

#include <memory>

#include <vulkan/vulkan.h>

#include "pygs/engine/vulkan/context.h"
#include "pygs/engine/vulkan/pipeline_layout.h"

namespace pygs {
namespace vk {

struct ComputePipelineCreateInfo {
  PipelineLayout layout;
  std::string compute_shader;
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
