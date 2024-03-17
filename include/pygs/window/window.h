#ifndef PYGS_WINDOW_WINDOW_H
#define PYGS_WINDOW_WINDOW_H

#include <memory>
#include <vector>

#include <pygs/window/event.h>

struct GLFWwindow;

namespace pygs {

class Window {
 public:
  Window();
  ~Window();

  GLFWwindow* window() const noexcept;

  bool ShouldClose() const;

  std::vector<Event> PollEvents();

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace pygs

#endif  // PYGS_WINDOW_WINDOW_H
