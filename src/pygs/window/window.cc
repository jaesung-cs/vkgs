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

  static void MouseButtonCallback(GLFWwindow* window, int button, int action,
                                  int mods) {
    bool valid = true;
    MouseButton event_button;
    bool event_pressed;

    switch (button) {
      case GLFW_MOUSE_BUTTON_LEFT:
        event_button = MouseButton::LEFT;
        break;

      case GLFW_MOUSE_BUTTON_RIGHT:
        event_button = MouseButton::RIGHT;
        break;

      default:
        valid = false;
    }

    switch (action) {
      case GLFW_PRESS:
        event_pressed = true;
        break;

      default:
        event_pressed = false;
    }

    auto* w = reinterpret_cast<Impl*>(glfwGetWindowUserPointer(window));
    w->MouseButtonClicked(event_button, event_pressed);
  }

 public:
  Impl() {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(width_, height_, "pygs", NULL, NULL);
    glfwSetWindowUserPointer(window_, this);
    glfwSetCursorPosCallback(window_, CursorPosCallback);
    glfwSetMouseButtonCallback(window_, MouseButtonCallback);
  }

  ~Impl() = default;

  GLFWwindow* window() const noexcept { return window_; }

  WindowSize size() const {
    int width, height;
    glfwGetFramebufferSize(window_, &width, &height);

    WindowSize result;
    result.width = width;
    result.height = height;
    return result;
  }

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

  void MouseButtonClicked(MouseButton button, bool pressed) {
    Event event;
    event.type = EventType::MOUSE_CLICK;
    event.mouse_click.button = button;
    event.mouse_click.pressed = pressed;
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

GLFWwindow* Window::window() const noexcept { return impl_->window(); }

WindowSize Window::size() const { return impl_->size(); }

bool Window::ShouldClose() const { return impl_->ShouldClose(); }

std::vector<Event> Window::PollEvents() { return impl_->PollEvents(); }

}  // namespace pygs
