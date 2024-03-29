#ifndef PYGS_ENGINE_ENGINE_H
#define PYGS_ENGINE_ENGINE_H

#include <memory>

namespace pygs {

class Splats;

class Engine {
 public:
  Engine();
  ~Engine();

  void AddSplats(const Splats& splats);
  void Run();

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace pygs

#endif  // PYGS_ENGINE_ENGINE_H
