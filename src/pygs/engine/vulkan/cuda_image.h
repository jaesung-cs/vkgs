#ifndef PYGS_ENGINE_VULKAN_CUDA_IMAGE_H
#define PYGS_ENGINE_VULKAN_CUDA_IMAGE_H

#include <memory>

#include <vulkan/vulkan.h>

#include "context.h"

namespace pygs {
namespace vk {

class CudaImage {
 public:
  CudaImage();

  CudaImage(Context context, uint32_t width, uint32_t height);

  ~CudaImage();

  operator bool() const noexcept { return impl_ != nullptr; }

  VkImage image() const;
  uint32_t width() const;
  uint32_t height() const;
  void* map();
  const void* map() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_CUDA_IMAGE_H
