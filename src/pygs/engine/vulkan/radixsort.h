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

  // Sort, num elements indirect
  void Sort(VkCommandBuffer command_buffer, uint32_t frame_index,
            VkBuffer num_elements_buffer, VkBuffer values, VkBuffer indices);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGIE_VULKAN_RADIXSORT_H
