#include "vkgs/vulkan/framebuffer.h"

namespace vkgs {
namespace vk {

class Framebuffer::Impl {
 public:
  Impl() = delete;

  Impl(Context context, const FramebufferCreateInfo& create_info) : context_(context) {
    auto image_count = create_info.image_specs.size();
    std::vector<VkFramebufferAttachmentImageInfo> attachment_images(image_count);
    for (int i = 0; i < image_count; ++i) {
      attachment_images[i] = {VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO};
      attachment_images[i].usage = create_info.image_specs[i].usage;
      attachment_images[i].width = create_info.image_specs[i].width;
      attachment_images[i].height = create_info.image_specs[i].height;
      attachment_images[i].layerCount = 1;
      attachment_images[i].viewFormatCount = 1;
      attachment_images[i].pViewFormats = &create_info.image_specs[i].format;
    }

    VkFramebufferAttachmentsCreateInfo attachments_info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO};
    attachments_info.attachmentImageInfoCount = attachment_images.size();
    attachments_info.pAttachmentImageInfos = attachment_images.data();

    VkFramebufferCreateInfo framebuffer_info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer_info.pNext = &attachments_info;
    framebuffer_info.flags = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT;
    framebuffer_info.renderPass = create_info.render_pass;
    framebuffer_info.attachmentCount = image_count;
    framebuffer_info.width = create_info.width;
    framebuffer_info.height = create_info.height;
    framebuffer_info.layers = 1;
    vkCreateFramebuffer(context_.device(), &framebuffer_info, NULL, &framebuffer_);

    width_ = create_info.width;
    height_ = create_info.height;
  }

  ~Impl() { vkDestroyFramebuffer(context_.device(), framebuffer_, NULL); }

  operator VkFramebuffer() const noexcept { return framebuffer_; }
  auto width() const noexcept { return width_; }
  auto height() const noexcept { return height_; }

 private:
  Context context_;
  VkFramebuffer framebuffer_ = VK_NULL_HANDLE;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
};

Framebuffer::Framebuffer() = default;

Framebuffer::Framebuffer(Context context, const FramebufferCreateInfo& create_info)
    : impl_(std::make_shared<Impl>(context, create_info)) {}

Framebuffer::~Framebuffer() = default;

Framebuffer::operator bool() const { return impl_ != nullptr; }
Framebuffer::operator VkFramebuffer() const { return *impl_; }
uint32_t Framebuffer::width() const { return impl_->width(); }
uint32_t Framebuffer::height() const { return impl_->height(); }

}  // namespace vk
}  // namespace vkgs
