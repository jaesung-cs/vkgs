#ifndef VKGS_ENGINE_ENGINE_H
#define VKGS_ENGINE_ENGINE_H

#include <memory>
#include <string>

namespace vkgs {

class Splats;

class Engine {
 public:
  Engine();
  ~Engine();

  void LoadSplats(const std::string& ply_filepath);
  void LoadSplatsAsync(const std::string& ply_filepath);

  void Run();

  void Close();

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vkgs

#endif  // VKGS_ENGINE_ENGINE_H
