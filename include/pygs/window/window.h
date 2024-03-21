#ifndef PYGS_WINDOW_WINDOW_H
#define PYGS_WINDOW_WINDOW_H

#include <memory>
#include <vector>

#include <pygs/window/event.h>

struct GLFWwindow;

namespace pygs {

struct WindowSize {
  uint32_t width = 0;
  uint32_t height = 0;
};

class Window {
 public:
  Window();
  ~Window();

  GLFWwindow* window() const noexcept;

  WindowSize size() const;

  bool ShouldClose() const;

  std::vector<Event> PollEvents();

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace pygs

#endif  // PYGS_WINDOW_WINDOW_H
