#include <pygs/engine/engine.h>

#include <unordered_map>

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "vulkan/context.h"
#include "vulkan/swapchain.h"

namespace pygs {

class Engine::Impl {
 public:
  Impl() {}

  ~Impl() {}

  void Draw(Window window) {
    auto window_ptr = window.window();
    if (swapchains_.count(window_ptr) == 0) {
      swapchains_[window_ptr] = vk::Swapchain(context_, window_ptr);
    }
  }

 private:
  vk::Context context_;
  std::unordered_map<GLFWwindow*, vk::Swapchain> swapchains_;
};

Engine::Engine() : impl_(std::make_shared<Impl>()) {}

Engine::~Engine() {}

void Engine::Draw(Window window) { impl_->Draw(window); }

}  // namespace pygs
