#ifndef PYGS_ENGINE_VULKAN_SHADER_MODULE_H
#define PYGS_ENGINE_VULKAN_SHADER_MODULE_H

#include <string>

#include <vulkan/vulkan.h>

namespace pygs {
namespace vk {

class ShaderSource {
 public:
  ShaderSource() = default;

  template <size_t N>
  ShaderSource(const uint32_t (&source)[N])
      : source_(source), size_(sizeof(uint32_t) * N) {}

  template <size_t N>
  ShaderSource& operator=(const uint32_t (&source)[N]) {
    source_ = source;
    size_ = sizeof(uint32_t) * N;
    return *this;
  }

  // byte size
  VkDeviceSize size() const noexcept { return size_; }
  const uint32_t* data() const noexcept { return source_; }

 private:
  VkDeviceSize size_ = 0;
  const uint32_t* source_ = nullptr;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_MODULE_H
