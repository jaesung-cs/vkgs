#ifndef PYGS_ENGINE_ENGINE_H
#define PYGS_ENGINE_ENGINE_H

#include <memory>

#include <pygs/window/window.h>

namespace pygs {

class Camera;

class Engine {
 public:
  Engine();
  ~Engine();

  void Draw(Window window, const Camera& camera);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace pygs

#endif  // PYGS_ENGINE_ENGINE_H
