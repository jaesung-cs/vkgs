#include "pygs/engine/vulkan/graphics_pipeline.h"

#include "pygs/engine/vulkan/shader_module.h"

namespace pygs {
namespace vk {
class GraphicsPipeline::Impl {
 public:
  Impl() = delete;

  Impl(Context context, const GraphicsPipelineCreateInfo& create_info)
      : context_(context) {
    VkDevice device = context.device();

    VkShaderModule vertex_module = CreateShaderModule(
        device, VK_SHADER_STAGE_VERTEX_BIT, create_info.vertex_shader);
    VkShaderModule fragment_module = CreateShaderModule(
        device, VK_SHADER_STAGE_FRAGMENT_BIT, create_info.fragment_shader);

    std::vector<VkPipelineShaderStageCreateInfo> stages(2);
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex_module;
    stages[0].pName = "main";

    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment_module;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertex_input_state = {
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input_state.vertexBindingDescriptionCount =
        create_info.input_bindings.size();
    vertex_input_state.pVertexBindingDescriptions =
        create_info.input_bindings.data();
    vertex_input_state.vertexAttributeDescriptionCount =
        create_info.input_attributes.size();
    vertex_input_state.pVertexAttributeDescriptions =
        create_info.input_attributes.data();

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state = {
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    input_assembly_state.topology = create_info.topology;
    input_assembly_state.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewport_state = {
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization_state = {
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization_state.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization_state.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterization_state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization_state.lineWidth = 1.f;

    VkPipelineMultisampleStateCreateInfo multisample_state = {
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample_state.rasterizationSamples = create_info.samples;

    VkPipelineDepthStencilStateCreateInfo depth_stencil_state = {
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth_stencil_state.depthTestEnable = create_info.depth_test;
    depth_stencil_state.depthWriteEnable = create_info.depth_write;
    depth_stencil_state.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendStateCreateInfo color_blend_state = {
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    color_blend_state.attachmentCount =
        create_info.color_blend_attachments.size();
    color_blend_state.pAttachments = create_info.color_blend_attachments.data();

    std::vector<VkDynamicState> dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,
                                                  VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_state = {
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic_state.dynamicStateCount = dynamic_states.size();
    dynamic_state.pDynamicStates = dynamic_states.data();

    VkGraphicsPipelineCreateInfo pipeline_info = {
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline_info.stageCount = stages.size();
    pipeline_info.pStages = stages.data();
    pipeline_info.pVertexInputState = &vertex_input_state;
    pipeline_info.pInputAssemblyState = &input_assembly_state;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterization_state;
    pipeline_info.pMultisampleState = &multisample_state;
    pipeline_info.pDepthStencilState = &depth_stencil_state;
    pipeline_info.pColorBlendState = &color_blend_state;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = create_info.layout;
    pipeline_info.renderPass = create_info.render_pass;
    pipeline_info.subpass = create_info.subpass;
    vkCreateGraphicsPipelines(device, NULL, 1, &pipeline_info, NULL,
                              &pipeline_);

    vkDestroyShaderModule(device, vertex_module, NULL);
    vkDestroyShaderModule(device, fragment_module, NULL);
  }

  ~Impl() { vkDestroyPipeline(context_.device(), pipeline_, NULL); }

  operator VkPipeline() const noexcept { return pipeline_; }

 private:
  Context context_;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

GraphicsPipeline::GraphicsPipeline() = default;

GraphicsPipeline::GraphicsPipeline(
    Context context, const GraphicsPipelineCreateInfo& create_info)
    : impl_(std::make_shared<Impl>(context, create_info)) {}

GraphicsPipeline::~GraphicsPipeline() = default;

GraphicsPipeline::operator VkPipeline() const { return *impl_; }

}  // namespace vk
}  // namespace pygs
