#ifndef VKGS_ENGINE_VULKAN_UNIFORM_BUFFER_H
#define VKGS_ENGINE_VULKAN_UNIFORM_BUFFER_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"

namespace vkgs {
namespace vk {

class UniformBufferBase {
 public:
  UniformBufferBase();

  UniformBufferBase(Context context, VkDeviceSize size);

  virtual ~UniformBufferBase();

  operator VkBuffer() const;

 protected:
  void* ptr();
  const void* ptr() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

template <typename T>
class UniformBuffer : public UniformBufferBase {
 public:
  UniformBuffer() {}

  UniformBuffer(Context context, size_t size)
      : UniformBufferBase(context, size * sizeof(T)) {}

  ~UniformBuffer() override {}

  T& operator[](size_t index) { return reinterpret_cast<T*>(ptr())[index]; }
  const T& operator[](size_t index) const {
    return reinterpret_cast<T*>(ptr())[index];
  }

  VkDeviceSize offset(size_t index) { return sizeof(T) * index; }

  VkDeviceSize element_size() { return sizeof(T); }

 private:
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_UNIFORM_BUFFER_H
