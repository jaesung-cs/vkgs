#include "render_pass.h"

namespace pygs {
namespace vk {

class RenderPass::Impl {
 public:
  Impl() = delete;

  Impl(Context context) : context_(context) {
    std::vector<VkAttachmentDescription2> attachments(3);
    attachments[0] = {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2};
    attachments[0].format = VK_FORMAT_B8G8R8A8_UNORM;
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
    attachments[2].format = VK_FORMAT_B8G8R8A8_UNORM;
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

  ~Impl() { vkDestroyRenderPass(context_.device(), render_pass_, NULL); }

  operator VkRenderPass() const noexcept { return render_pass_; }

 private:
  Context context_;
  VkRenderPass render_pass_ = VK_NULL_HANDLE;
};

RenderPass::RenderPass() = default;

RenderPass::RenderPass(Context context)
    : impl_(std::make_shared<Impl>(context)) {}

RenderPass::~RenderPass() = default;

RenderPass::operator VkRenderPass() const { return *impl_; }

}  // namespace vk
}  // namespace pygs
