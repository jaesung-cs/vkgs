#ifndef PYGS_ENGINE_SPLAT_LOAD_THREAD_H
#define PYGS_ENGINE_SPLAT_LOAD_THREAD_H

#include <memory>
#include <string>

#include "pygs/engine/vulkan/context.h"
#include "pygs/engine/vulkan/buffer.h"

namespace pygs {

class SplatLoadThread {
 private:
  struct Progress {
    uint32_t total_point_count = 0;
    uint32_t loaded_point_count = 0;
  };

 public:
  SplatLoadThread();

  SplatLoadThread(vk::Context context);

  ~SplatLoadThread();

  void Start(const std::string& ply_filepath, vk::Buffer position,
             vk::Buffer cov3d, vk::Buffer sh, vk::Buffer opacity);

  Progress progress();

  void cancel();

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace pygs

#endif  // PYGS_ENGINE_SPLAT_LOAD_THREAD_H
