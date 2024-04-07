#include "pygs/engine/vulkan/compute_pipeline.h"

#include "pygs/engine/vulkan/shader_module.h"

namespace pygs {
namespace vk {

class ComputePipeline::Impl {
 public:
  Impl() = delete;

  Impl(Context context, const ComputePipelineCreateInfo& create_info)
      : context_(context) {
    VkShaderModule compute_module =
        CreateShaderModule(context_.device(), VK_SHADER_STAGE_COMPUTE_BIT,
                           create_info.compute_shader);

    VkPipelineShaderStageCreateInfo stage_info = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = compute_module;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = stage_info;
    pipeline_info.layout = create_info.layout;
    vkCreateComputePipelines(context_.device(), VK_NULL_HANDLE, 1,
                             &pipeline_info, NULL, &pipeline_);

    vkDestroyShaderModule(context_.device(), compute_module, NULL);
  }

  ~Impl() { vkDestroyPipeline(context_.device(), pipeline_, NULL); }

  operator VkPipeline() const noexcept { return pipeline_; }

 private:
  Context context_;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

ComputePipeline::ComputePipeline() = default;

ComputePipeline::ComputePipeline(Context context,
                                 const ComputePipelineCreateInfo& create_info)
    : impl_(std::make_shared<Impl>(context, create_info)) {}

ComputePipeline::~ComputePipeline() = default;

ComputePipeline::operator VkPipeline() const { return *impl_; }

}  // namespace vk
}  // namespace pygs
