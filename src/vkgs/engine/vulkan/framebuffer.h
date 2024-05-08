#ifndef VKGS_ENGINE_VULKAN_FRAMEBUFFER_H
#define VKGS_ENGINE_VULKAN_FRAMEBUFFER_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/image_spec.h"

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

  operator VkFramebuffer() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_FRAMEBUFFER_H
