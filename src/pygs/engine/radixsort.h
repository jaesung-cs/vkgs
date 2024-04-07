#ifndef PYGS_ENGIE_VULKAN_RADIXSORT_H
#define PYGS_ENGIE_VULKAN_RADIXSORT_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vulkan/context.h"

namespace pygs {

class Radixsort {
 public:
  Radixsort();

  Radixsort(vk::Context context, size_t max_num_elements);

  ~Radixsort();

  // Sort, num elements indirect
  void Sort(VkCommandBuffer command_buffer, uint32_t frame_index,
            VkBuffer num_elements_buffer, VkBuffer values, VkBuffer indices);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace pygs

#endif  // PYGS_ENGIE_VULKAN_RADIXSORT_H
