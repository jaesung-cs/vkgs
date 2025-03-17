#include "vkgs/vulkan/timestamp_query.h"

#include <iostream>

namespace vkgs {
namespace vk {

class TimestampQuery::Impl {
 public:
  Impl() = delete;

  Impl(Context context, uint32_t size) : context_(context), size_(size) {
    VkQueryPoolCreateInfo query_pool_info = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_pool_info.queryCount = size;
    vkCreateQueryPool(context_.device(), &query_pool_info, NULL, &query_pool_);
  }

  ~Impl() { vkDestroyQueryPool(context_.device(), query_pool_, NULL); }

  operator VkQueryPool() const noexcept { return query_pool_; }

  std::vector<uint64_t> timestamps() const {
    if (value_ > 0) {
      std::vector<uint64_t> result(value_);
      vkGetQueryPoolResults(context_.device(), query_pool_, 0, value_, value_ * sizeof(uint64_t), result.data(),
                            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
      return result;
    }
    return {};
  }

  void reset(VkCommandBuffer command_buffer) {
    vkCmdResetQueryPool(command_buffer, query_pool_, 0, size_);
    value_ = 0;
  }

  void write(VkCommandBuffer command_buffer, VkPipelineStageFlagBits stage) {
    vkCmdWriteTimestamp(command_buffer, stage, query_pool_, value_);
    value_++;
  }

 private:
  Context context_;
  uint32_t size_ = 0;
  VkQueryPool query_pool_ = VK_NULL_HANDLE;
  uint32_t value_ = 0;
};

TimestampQuery::TimestampQuery() = default;

TimestampQuery::TimestampQuery(Context context, uint32_t size) : impl_(std::make_shared<Impl>(context, size)) {}

TimestampQuery::~TimestampQuery() = default;

TimestampQuery::operator bool() const { return impl_ != nullptr; }

TimestampQuery::operator VkQueryPool() const { return *impl_; }

std::vector<uint64_t> TimestampQuery::timestamps() const { return impl_->timestamps(); }

void TimestampQuery::reset(VkCommandBuffer command_buffer) { impl_->reset(command_buffer); }

void TimestampQuery::write(VkCommandBuffer command_buffer, VkPipelineStageFlagBits stage) {
  impl_->write(command_buffer, stage);
}

}  // namespace vk
}  // namespace vkgs
