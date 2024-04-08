#ifndef PYGS_ENGINE_INTEROP_SEMAPHORE_H
#define PYGS_ENGINE_INTEROP_SEMAPHORE_H

#include <cuda_runtime.h>

#include "pygs/engine/vulkan/context.h"

namespace pygs {
namespace vk {

class CudaSemaphore {
 public:
  CudaSemaphore();

  explicit CudaSemaphore(Context context);

  ~CudaSemaphore();

  VkSemaphore semaphore() const;
  void signal(cudaStream_t stream);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_INTEROP_SEMAPHORE_H
