#ifndef VKGS_VULKAN_ATTACHMENT_H
#define VKGS_VULKAN_ATTACHMENT_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/vulkan/context.h"
#include "vkgs/vulkan/image_spec.h"

namespace vkgs {
namespace vk {

class Context;

class Attachment {
 public:
  Attachment();

  Attachment(Context context, uint32_t width, uint32_t height, VkFormat format, VkSampleCountFlagBits samples,
             bool input_attachment);

  ~Attachment();

  operator VkImageView() const;

  VkImage image() const;
  VkImageUsageFlags usage() const;
  VkFormat format() const;
  ImageSpec image_spec() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_ATTACHMENT_H
