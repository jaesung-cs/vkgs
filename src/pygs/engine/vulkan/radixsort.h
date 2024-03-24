#ifndef PYGS_ENGIE_VULKAN_RADIXSORT_H
#define PYGS_ENGIE_VULKAN_RADIXSORT_H

#include <memory>

#include <vulkan/vulkan.h>

#include "context.h"

namespace pygs {
namespace vk {

class Radixsort {
 public:
  Radixsort();

  Radixsort(Context context, size_t max_num_elements);

  ~Radixsort();

  void Sort(VkCommandBuffer command_buffer, uint32_t frame_index,
            size_t num_elements, VkBuffer values, VkBuffer indices);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGIE_VULKAN_RADIXSORT_H
