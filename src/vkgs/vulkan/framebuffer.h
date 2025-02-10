#ifndef VKGS_VULKAN_FRAMEBUFFER_H
#define VKGS_VULKAN_FRAMEBUFFER_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/vulkan/context.h"
#include "vkgs/vulkan/image_spec.h"

namespace vkgs {
namespace vk {

struct FramebufferCreateInfo {
  VkRenderPass render_pass = VK_NULL_HANDLE;
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<ImageSpec> image_specs;
};

class Framebuffer {
 public:
  Framebuffer();

  Framebuffer(Context context, const FramebufferCreateInfo& create_info);

  ~Framebuffer();

  operator bool() const;
  operator VkFramebuffer() const;
  uint32_t width() const;
  uint32_t height() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_FRAMEBUFFER_H
