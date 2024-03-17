#include <pygs/engine/engine.h>

namespace pygs {

class Engine::Impl {
 public:
  Impl() {}

  ~Impl() {}

  void Draw(Window window) {}

 private:
};

Engine::Engine() : impl_(std::make_shared<Impl>()) {}

Engine::~Engine() {}

void Engine::Draw(Window window) { impl_->Draw(window); }

}  // namespace pygs
