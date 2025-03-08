#include "vkgs/vulkan/descriptor.h"

namespace vkgs {
namespace vk {

class Descriptor::Impl {
 public:
  Impl() = delete;

  Impl(Context context, DescriptorLayout layout) : context_(context), layout_(layout) {
    VkDescriptorSetLayout layout_handle = layout;

    VkDescriptorSetAllocateInfo descriptor_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    descriptor_info.descriptorPool = context.descriptor_pool();
    descriptor_info.descriptorSetCount = 1;
    descriptor_info.pSetLayouts = &layout_handle;
    vkAllocateDescriptorSets(context.device(), &descriptor_info, &descriptor_);
  }

  ~Impl() {}

  operator VkDescriptorSet() const noexcept { return descriptor_; }

  void Update(uint32_t binding, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size) {
    VkDescriptorBufferInfo buffer_info = {};
    buffer_info.buffer = buffer;
    buffer_info.offset = offset;
    buffer_info.range = size > 0 ? size : VK_WHOLE_SIZE;

    VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptor_;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = layout_.type(binding);
    write.pBufferInfo = &buffer_info;
    vkUpdateDescriptorSets(context_.device(), 1, &write, 0, NULL);
  }

  void UpdateInputAttachment(uint32_t binding, VkImageView image_view) {
    VkDescriptorImageInfo image_info = {};
    image_info.sampler = VK_NULL_HANDLE;
    image_info.imageView = image_view;
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptor_;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = layout_.type(binding);
    write.pImageInfo = &image_info;
    vkUpdateDescriptorSets(context_.device(), 1, &write, 0, NULL);
  }

 private:
  Context context_;
  DescriptorLayout layout_;
  VkDescriptorSet descriptor_ = VK_NULL_HANDLE;
};

Descriptor::Descriptor() = default;

Descriptor::Descriptor(Context context, DescriptorLayout layout) : impl_(std::make_shared<Impl>(context, layout)) {}

Descriptor::~Descriptor() = default;

Descriptor::operator VkDescriptorSet() const { return *impl_; }

void Descriptor::Update(uint32_t binding, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size) {
  impl_->Update(binding, buffer, offset, size);
}

void Descriptor::UpdateInputAttachment(uint32_t binding, VkImageView image_view) {
  impl_->UpdateInputAttachment(binding, image_view);
}

}  // namespace vk
}  // namespace vkgs
