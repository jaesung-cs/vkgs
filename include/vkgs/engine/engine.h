#ifndef VKGS_ENGINE_ENGINE_H
#define VKGS_ENGINE_ENGINE_H

#include <memory>
#include <string>

namespace vkgs {

class Engine {
 public:
  Engine();
  ~Engine();

  void Run(const std::string& plyFilepath);

  void Close();

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vkgs

#endif  // VKGS_ENGINE_ENGINE_H
