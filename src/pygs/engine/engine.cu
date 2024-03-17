#include <pygs/engine/engine.h>

#include <iostream>
#include <unordered_map>

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "vulkan/context.h"
#include "vulkan/cuda_image.h"
#include "vulkan/swapchain.h"

namespace pygs {
namespace {}

class Engine::Impl {
 public:
  Impl() {}

  ~Impl() {}

  void Draw(Window window) {
    auto window_ptr = window.window();
    if (swapchains_.count(window_ptr) == 0) {
      swapchains_[window_ptr] = vk::Swapchain(context_, window_ptr);
    }

    auto swapchain = swapchains_[window_ptr];

    if (!cuda_image1_ || cuda_image1_.width() != swapchain.width() ||
        cuda_image1_.height() != swapchain.height()) {
      cuda_image1_ =
          vk::CudaImage(context_, swapchain.width(), swapchain.height());
    }

    if (!cuda_image2_ || cuda_image2_.width() != swapchain.width() ||
        cuda_image2_.height() != swapchain.height()) {
      cuda_image2_ =
          vk::CudaImage(context_, swapchain.width(), swapchain.height());
    }
  }

 private:
  vk::Context context_;
  std::unordered_map<GLFWwindow*, vk::Swapchain> swapchains_;
  vk::CudaImage cuda_image1_;
  vk::CudaImage cuda_image2_;
};

Engine::Engine() : impl_(std::make_shared<Impl>()) {}

Engine::~Engine() {}

void Engine::Draw(Window window) { impl_->Draw(window); }

}  // namespace pygs
