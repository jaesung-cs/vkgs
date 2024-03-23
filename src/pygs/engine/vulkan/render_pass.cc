#include "render_pass.h"

namespace pygs {
namespace vk {

class RenderPass::Impl {
 public:
  Impl() = delete;

  Impl(Context context, RenderPassType type) : context_(context) {
    switch (type) {
      case RenderPassType::NORMAL:
        CreateRenderPassNormal();
        break;
      case RenderPassType::OIT:
        CreateRenderPassOit();
        break;
    }
  }

  ~Impl() { vkDestroyRenderPass(context_.device(), render_pass_, NULL); }

  operator VkRenderPass() const noexcept { return render_pass_; }

 private:
  void CreateRenderPassNormal() {
    std::vector<VkAttachmentDescription2> attachments(3);
    attachments[0] = {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2};
    attachments[0].format = VK_FORMAT_B8G8R8A8_SRGB;
    attachments[0].samples = VK_SAMPLE_COUNT_4_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    attachments[1] = {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2};
    attachments[1].format = VK_FORMAT_D24_UNORM_S8_UINT;
    attachments[1].samples = VK_SAMPLE_COUNT_4_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    attachments[2] = {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2};
    attachments[2].format = VK_FORMAT_B8G8R8A8_SRGB;
    attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[2].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    std::vector<VkAttachmentReference2> pass0_colors(1);
    pass0_colors[0] = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    pass0_colors[0].attachment = 0;
    pass0_colors[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    pass0_colors[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    VkAttachmentReference2 pass0_depth = {
        VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    pass0_depth.attachment = 1;
    pass0_depth.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    pass0_depth.aspectMask =
        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;

    VkAttachmentReference2 pass0_resolve = {
        VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    pass0_resolve.attachment = 2;
    pass0_resolve.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    pass0_resolve.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    std::vector<VkSubpassDescription2> subpasses(1);
    subpasses[0] = {VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2};
    subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses[0].colorAttachmentCount = pass0_colors.size();
    subpasses[0].pColorAttachments = pass0_colors.data();
    subpasses[0].pResolveAttachments = &pass0_resolve;
    subpasses[0].pDepthStencilAttachment = &pass0_depth;

    std::vector<VkSubpassDependency2> dependencies(1);
    dependencies[0] = {VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].dstStageMask =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask =
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dstAccessMask =
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo2 render_pass_info = {
        VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2};
    render_pass_info.attachmentCount = attachments.size();
    render_pass_info.pAttachments = attachments.data();
    render_pass_info.subpassCount = subpasses.size();
    render_pass_info.pSubpasses = subpasses.data();
    render_pass_info.dependencyCount = dependencies.size();
    render_pass_info.pDependencies = dependencies.data();
    vkCreateRenderPass2(context_.device(), &render_pass_info, NULL,
                        &render_pass_);
  }

  void CreateRenderPassOit() {
    std::vector<VkAttachmentDescription2> attachments(5);
    // depth
    attachments[0] = {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2};
    attachments[0].format = VK_FORMAT_D24_UNORM_S8_UINT;
    attachments[0].samples = VK_SAMPLE_COUNT_4_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // opaque color
    attachments[1] = {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2};
    attachments[1].format = VK_FORMAT_B8G8R8A8_SRGB;
    attachments[1].samples = VK_SAMPLE_COUNT_4_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // transparent color
    // 16-bit doesn't work for GS settings, due to precision
    attachments[2] = {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2};
    attachments[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attachments[2].samples = VK_SAMPLE_COUNT_4_BIT;
    attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[2].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // prod alpha
    attachments[3] = {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2};
    attachments[3].format = VK_FORMAT_R32_SFLOAT;
    attachments[3].samples = VK_SAMPLE_COUNT_4_BIT;
    attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[3].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[3].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // resolve color
    attachments[4] = {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2};
    attachments[4].format = VK_FORMAT_B8G8R8A8_SRGB;
    attachments[4].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[4].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[4].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[4].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[4].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[4].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[4].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    std::vector<VkSubpassDescription2> subpasses(3);
    // pass 0
    std::vector<VkAttachmentReference2> pass0_colors(1);
    pass0_colors[0] = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    pass0_colors[0].attachment = 1;
    pass0_colors[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    pass0_colors[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    VkAttachmentReference2 pass0_depth = {
        VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    pass0_depth.attachment = 0;
    pass0_depth.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    pass0_depth.aspectMask =
        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;

    subpasses[0] = {VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2};
    subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses[0].colorAttachmentCount = pass0_colors.size();
    subpasses[0].pColorAttachments = pass0_colors.data();
    subpasses[0].pDepthStencilAttachment = &pass0_depth;

    // pass 1
    std::vector<VkAttachmentReference2> pass1_colors(2);
    pass1_colors[0] = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    pass1_colors[0].attachment = 2;
    pass1_colors[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    pass1_colors[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    pass1_colors[1] = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    pass1_colors[1].attachment = 3;
    pass1_colors[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    pass1_colors[1].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    VkAttachmentReference2 pass1_depth = {
        VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    pass1_depth.attachment = 0;
    pass1_depth.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    pass1_depth.aspectMask =
        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;

    std::vector<uint32_t> pass1_preserve = {1};

    subpasses[1] = {VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2};
    subpasses[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses[1].colorAttachmentCount = pass1_colors.size();
    subpasses[1].pColorAttachments = pass1_colors.data();
    subpasses[1].pDepthStencilAttachment = &pass1_depth;
    subpasses[1].preserveAttachmentCount = pass1_preserve.size();
    subpasses[1].pPreserveAttachments = pass1_preserve.data();

    // pass 2
    std::vector<VkAttachmentReference2> pass2_inputs(2);
    pass2_inputs[0] = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    pass2_inputs[0].attachment = 2;
    pass2_inputs[0].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    pass2_inputs[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    pass2_inputs[1] = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    pass2_inputs[1].attachment = 3;
    pass2_inputs[1].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    pass2_inputs[1].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    std::vector<VkAttachmentReference2> pass2_colors(1);
    pass2_colors[0] = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    pass2_colors[0].attachment = 1;
    pass2_colors[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    pass2_colors[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    VkAttachmentReference2 pass2_resolve = {
        VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    pass2_resolve.attachment = 4;
    pass2_resolve.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    pass2_resolve.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    subpasses[2] = {VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2};
    subpasses[2].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses[2].inputAttachmentCount = pass2_inputs.size();
    subpasses[2].pInputAttachments = pass2_inputs.data();
    subpasses[2].colorAttachmentCount = pass2_colors.size();
    subpasses[2].pColorAttachments = pass2_colors.data();
    subpasses[2].pResolveAttachments = &pass2_resolve;

    std::vector<VkSubpassDependency2> dependencies(5);
    // EXTERNAL -> 0, attachment 0 & 1, DEPTH & COLOR R/W -> DEPTH & COLOR R/W
    dependencies[0] = {VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].dstStageMask =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask =
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dstAccessMask =
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;

    // EXTERNAL -> 1, attachment 2 & 3, INPUT -> COLOR R/W
    dependencies[1] = {VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2};
    dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].dstSubpass = 1;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    dependencies[1].dstStageMask =
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;

    // EXTERNAL -> 2, attachment 4, COLOR R/W -> COLOR R/W
    dependencies[2] = {VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2};
    dependencies[2].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[2].dstSubpass = 2;
    dependencies[2].srcStageMask =
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[2].dstStageMask =
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[2].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[2].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;

    // 0 -> 1, attachment 0, DEPTH R/W -> DEPTH R
    dependencies[3] = {VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2};
    dependencies[3].srcSubpass = 0;
    dependencies[3].dstSubpass = 1;
    dependencies[3].srcStageMask =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    dependencies[3].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[3].srcAccessMask =
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[3].dstAccessMask =
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

    // 1 -> 2, attachment 2, 3, COLOR W -> INPUT R
    dependencies[4] = {VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2};
    dependencies[4].srcSubpass = 1;
    dependencies[4].dstSubpass = 2;
    dependencies[4].srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[4].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    dependencies[4].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[4].dstAccessMask = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT;

    VkRenderPassCreateInfo2 render_pass_info = {
        VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2};
    render_pass_info.attachmentCount = attachments.size();
    render_pass_info.pAttachments = attachments.data();
    render_pass_info.subpassCount = subpasses.size();
    render_pass_info.pSubpasses = subpasses.data();
    render_pass_info.dependencyCount = dependencies.size();
    render_pass_info.pDependencies = dependencies.data();
    vkCreateRenderPass2(context_.device(), &render_pass_info, NULL,
                        &render_pass_);
  }

  Context context_;
  VkRenderPass render_pass_ = VK_NULL_HANDLE;
};

RenderPass::RenderPass() = default;

RenderPass::RenderPass(Context context, RenderPassType type)
    : impl_(std::make_shared<Impl>(context, type)) {}

RenderPass::~RenderPass() = default;

RenderPass::operator VkRenderPass() const { return *impl_; }

}  // namespace vk
}  // namespace pygs
