#ifndef PYGS_ENGINE_VULKAN_RENDER_PASS_H
#define PYGS_ENGINE_VULKAN_RENDER_PASS_H

#include <memory>

#include <vulkan/vulkan.h>

#include "pygs/engine/vulkan/context.h"

namespace pygs {
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
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_RENDER_PASS_H
