#ifndef PYGS_ENGINE_VULKAN_CUDA_SEMAPHORE_H
#define PYGS_ENGINE_VULKAN_CUDA_SEMAPHORE_H

#include "context.h"

#include <cuda_runtime.h>

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

#endif  // PYGS_ENGINE_VULKAN_CUDA_SEMAPHORE_H
