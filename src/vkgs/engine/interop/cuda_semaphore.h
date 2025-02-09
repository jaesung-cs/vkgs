#ifndef VKGS_ENGINE_INTEROP_SEMAPHORE_H
#define VKGS_ENGINE_INTEROP_SEMAPHORE_H

#include <cuda_runtime.h>

#include "vkgs/vulkan/context.h"

namespace vkgs {
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
}  // namespace vkgs

#endif  // VKGS_ENGINE_INTEROP_SEMAPHORE_H
