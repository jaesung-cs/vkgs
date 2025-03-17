#ifndef VKGS_VULKAN_TIMSETAMP_QUERY_H
#define VKGS_VULKAN_TIMSETAMP_QUERY_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/vulkan/context.h"

namespace vkgs {
namespace vk {

class TimestampQuery {
 public:
  TimestampQuery();

  explicit TimestampQuery(Context context, uint32_t size);

  ~TimestampQuery();

  operator bool() const;
  operator VkQueryPool() const;

  std::vector<uint64_t> timestamps() const;

  void reset(VkCommandBuffer command_buffer);
  void write(VkCommandBuffer command_buffer, VkPipelineStageFlagBits stage);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_TIMSETAMP_QUERY_H
