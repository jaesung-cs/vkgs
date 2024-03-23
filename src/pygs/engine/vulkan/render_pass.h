#ifndef PYGS_ENGINE_VULKAN_RENDER_PASS_H
#define PYGS_ENGINE_VULKAN_RENDER_PASS_H

#include <memory>

#include <vulkan/vulkan.h>

#include "context.h"

namespace pygs {
namespace vk {

enum class RenderPassType {
  NORMAL,
  OIT,
};

class RenderPass {
 public:
  RenderPass();

  RenderPass(Context context, RenderPassType type);

  ~RenderPass();

  operator VkRenderPass() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_RENDER_PASS_H
