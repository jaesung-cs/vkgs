#ifndef PYGS_ENGINE_ENGINE_H
#define PYGS_ENGINE_ENGINE_H

#include <memory>
#include <string>

namespace pygs {

class Splats;

class Engine {
 public:
  Engine();
  ~Engine();

  void LoadSplats(const std::string& ply_filepath);
  void Run();

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace pygs

#endif  // PYGS_ENGINE_ENGINE_H
