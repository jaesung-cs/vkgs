#ifndef PYGS_ENGINE_VULKAN_SWAPCHAIN_H
#define PYGS_ENGINE_VULKAN_SWAPCHAIN_H

#include <memory>

#include <vulkan/vulkan.h>

#include "context.h"

struct GLFWwindow;

namespace pygs {
namespace vk {

class Swapchain {
 public:
  Swapchain();

  Swapchain(Context context, GLFWwindow* window);

  ~Swapchain();

  VkSwapchainKHR swapchain() const;
  uint32_t width() const;
  uint32_t height() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SWAPCHAIN_H
