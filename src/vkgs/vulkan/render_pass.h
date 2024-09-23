#ifndef VKGS_VULKAN_RENDER_PASS_H
#define VKGS_VULKAN_RENDER_PASS_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/vulkan/context.h"

namespace vkgs {
namespace vk {

enum class RenderPassType {
  NORMAL,
  OIT,
};

class RenderPass {
 public:
  RenderPass();

  RenderPass(Context context, VkSampleCountFlagBits samples,
             VkFormat depth_format);

  ~RenderPass();

  operator VkRenderPass() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_RENDER_PASS_H
