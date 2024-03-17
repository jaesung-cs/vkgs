#include <pygs/window/window.h>

#include <stdexcept>

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace pygs {

class Window::Impl {
 public:
  static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    auto* w = reinterpret_cast<Impl*>(glfwGetWindowUserPointer(window));
    w->CursorPos(xpos, ypos);
  }

 public:
  Impl() {
    if (glfwInit() == GLFW_FALSE)
      throw std::runtime_error("Failed to initialize glfw.");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(width_, height_, "pygs", NULL, NULL);
    glfwSetWindowUserPointer(window_, this);
    glfwSetCursorPosCallback(window_, CursorPosCallback);
  }

  ~Impl() { glfwTerminate(); }

  bool ShouldClose() const { return glfwWindowShouldClose(window_); }

  std::vector<Event> PollEvents() {
    events_.clear();
    glfwPollEvents();
    return events_;
  }

  void CursorPos(double x, double y) {
    Event event;
    event.type = EventType::MOUSE_MOVE;
    event.mouse_move.x = x;
    event.mouse_move.y = y;
    events_.push_back(event);
  }

 private:
  int width_ = 1600;
  int height_ = 900;
  GLFWwindow* window_ = NULL;

  std::vector<Event> events_;
};

Window::Window() : impl_(std::make_shared<Impl>()) {}

Window::~Window() {}

bool Window::ShouldClose() const { return impl_->ShouldClose(); }

std::vector<Event> Window::PollEvents() { return impl_->PollEvents(); }

}  // namespace pygs
