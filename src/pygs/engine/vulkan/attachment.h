#ifndef PYGS_ENGINE_VULKAN_ATTACHMENT_H
#define PYGS_ENGINE_VULKAN_ATTACHMENT_H

#include <memory>

#include <vulkan/vulkan.h>

namespace pygs {
namespace vk {

class Context;

class Attachment {
 public:
  Attachment();

  Attachment(Context context, VkFormat format, VkSampleCountFlagBits samples);

  ~Attachment();

  operator VkImageView() const;

  VkImage image() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_ATTACHMENT_H
