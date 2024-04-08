#ifndef PYGS_ENGINE_ENGINE_H
#define PYGS_ENGINE_ENGINE_H

#include <memory>
#include <future>

namespace pygs {

class Splats;

class Engine {
 public:
  Engine();
  ~Engine();

  void AddSplats(const Splats& splats);
  void AddSplatsAsync(std::future<Splats>&& splats_future);
  void Run();

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace pygs

#endif  // PYGS_ENGINE_ENGINE_H
