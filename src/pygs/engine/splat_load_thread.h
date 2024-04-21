#ifndef PYGS_ENGINE_SPLAT_LOAD_THREAD_H
#define PYGS_ENGINE_SPLAT_LOAD_THREAD_H

#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "pygs/engine/vulkan/context.h"
#include "pygs/engine/vulkan/buffer.h"

namespace pygs {

class SplatLoadThread {
 private:
  struct Progress {
    uint32_t total_point_count = 0;
    uint32_t loaded_point_count = 0;

    vk::Buffer ply_buffer;

    // buffer barriers by load thread from previous to current progress() call.
    // this must be consumed by receiving thread.
    std::vector<VkBufferMemoryBarrier2> buffer_barriers;
  };

 public:
  SplatLoadThread();

  SplatLoadThread(vk::Context context);

  ~SplatLoadThread();

  void Start(const std::string& ply_filepath);

  Progress progress();

  void cancel();

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace pygs

#endif  // PYGS_ENGINE_SPLAT_LOAD_THREAD_H
