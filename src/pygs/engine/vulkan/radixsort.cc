#include "radixsort.h"

#include "descriptor_layout.h"
#include "pipeline_layout.h"
#include "descriptor.h"
#include "compute_pipeline.h"
#include "buffer.h"

#include "shader/multi_radixsort.h"

namespace pygs {
namespace vk {

class Radixsort::Impl {
 public:
  Impl() = delete;

  Impl(Context context, size_t max_num_elements)
      : context_(context), max_num_elements_(max_num_elements) {
    DescriptorLayoutCreateInfo descriptor_layout_info = {};
    descriptor_layout_info.bindings.resize(5);
    descriptor_layout_info.bindings[0].binding = 0;
    descriptor_layout_info.bindings[0].descriptor_type =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[0].stage_flags =
        VK_SHADER_STAGE_COMPUTE_BIT;
    descriptor_layout_info.bindings[1].binding = 1;
    descriptor_layout_info.bindings[1].descriptor_type =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[1].stage_flags =
        VK_SHADER_STAGE_COMPUTE_BIT;
    descriptor_layout_info.bindings[2].binding = 2;
    descriptor_layout_info.bindings[2].descriptor_type =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[2].stage_flags =
        VK_SHADER_STAGE_COMPUTE_BIT;
    descriptor_layout_info.bindings[3].binding = 3;
    descriptor_layout_info.bindings[3].descriptor_type =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[3].stage_flags =
        VK_SHADER_STAGE_COMPUTE_BIT;
    descriptor_layout_info.bindings[4].binding = 4;
    descriptor_layout_info.bindings[4].descriptor_type =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[4].stage_flags =
        VK_SHADER_STAGE_COMPUTE_BIT;
    descriptor_layout_ = DescriptorLayout(context_, descriptor_layout_info);

    descriptor_layout_info.bindings.resize(2);
    descriptor_layout_info.bindings[0].binding = 0;
    descriptor_layout_info.bindings[0].descriptor_type =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[0].stage_flags =
        VK_SHADER_STAGE_COMPUTE_BIT;
    descriptor_layout_info.bindings[1].binding = 1;
    descriptor_layout_info.bindings[1].descriptor_type =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[1].stage_flags =
        VK_SHADER_STAGE_COMPUTE_BIT;
    indirect_layout_ = DescriptorLayout(context_, descriptor_layout_info);

    // 2 frames, 2 ping-pong descriptors
    descriptors_.resize(4);
    for (int i = 0; i < 4; i++) {
      descriptors_[i] = Descriptor(context_, descriptor_layout_);
    }

    indirect_descriptors_.resize(2);
    for (int i = 0; i < 2; i++) {
      indirect_descriptors_[i] = Descriptor(context_, indirect_layout_);
    }

    PipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.layouts = {descriptor_layout_, indirect_layout_};
    pipeline_layout_info.push_constants.resize(1);
    pipeline_layout_info.push_constants[0].stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_layout_info.push_constants[0].offset = 0;
    pipeline_layout_info.push_constants[0].size =
        sizeof(shader::radixsort::PushConstants);
    pipeline_layout_ = PipelineLayout(context_, pipeline_layout_info);

    ComputePipelineCreateInfo pipeline_info;
    pipeline_info.layout = pipeline_layout_;
    pipeline_info.compute_shader = shader::multi_radixsort_indirect_comp;
    indirect_pipeline_ = ComputePipeline(context_, pipeline_info);
    pipeline_info.compute_shader = shader::multi_radixsort_histograms_comp;
    histogram_pipeline_ = ComputePipeline(context_, pipeline_info);
    pipeline_info.compute_shader = shader::multi_radixsort_comp;
    radixsort_pipeline_ = ComputePipeline(context_, pipeline_info);

    // Preallocate buffers
    indirect_buffer_ = Buffer(context_, 5 * sizeof(uint32_t),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    value_buffer_ = Buffer(context_, max_num_elements * sizeof(uint32_t),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    index_buffer_ = Buffer(context_, max_num_elements * sizeof(uint32_t),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    uint32_t globalInvocationSize = max_num_elements / NUM_BLOCKS_PER_WORKGROUP;
    uint32_t remainder = max_num_elements % NUM_BLOCKS_PER_WORKGROUP;
    globalInvocationSize += remainder > 0 ? 1 : 0;
    uint32_t NUMBER_OF_WORKGROUPS =
        (globalInvocationSize + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    histogram_buffer_ = Buffer(
        context_, NUMBER_OF_WORKGROUPS * RADIX_SORT_BINS * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  }

  ~Impl() {}

  void Sort(VkCommandBuffer command_buffer, uint32_t frame_index,
            VkBuffer num_elements_buffer, VkBuffer values, VkBuffer indices) {
    VkDeviceSize buffer_size = value_buffer_.size();
    VkDeviceSize histogram_buffer_size = histogram_buffer_.size();

    // update descriptors
    descriptors_[frame_index * 2 + 0].Update(0, values, 0, buffer_size);
    descriptors_[frame_index * 2 + 0].Update(1, value_buffer_, 0, buffer_size);
    descriptors_[frame_index * 2 + 0].Update(2, histogram_buffer_, 0,
                                             histogram_buffer_size);
    descriptors_[frame_index * 2 + 0].Update(3, indices, 0, buffer_size);
    descriptors_[frame_index * 2 + 0].Update(4, index_buffer_, 0, buffer_size);

    descriptors_[frame_index * 2 + 1].Update(0, value_buffer_, 0, buffer_size);
    descriptors_[frame_index * 2 + 1].Update(1, values, 0, buffer_size);
    descriptors_[frame_index * 2 + 1].Update(2, histogram_buffer_, 0,
                                             histogram_buffer_size);
    descriptors_[frame_index * 2 + 1].Update(3, index_buffer_, 0, buffer_size);
    descriptors_[frame_index * 2 + 1].Update(4, indices, 0, buffer_size);

    indirect_descriptors_[frame_index].Update(0, indirect_buffer_, 0,
                                              indirect_buffer_.size());
    indirect_descriptors_[frame_index].Update(1, num_elements_buffer, 0,
                                              4 * sizeof(uint32_t));

    // calculate indirect
    std::vector<VkBufferMemoryBarrier2> buffer_barriers(1);
    buffer_barriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    buffer_barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    buffer_barriers[0].srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    buffer_barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    buffer_barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    buffer_barriers[0].buffer = indirect_buffer_;
    buffer_barriers[0].offset = 0;
    buffer_barriers[0].size = indirect_buffer_.size();

    VkDependencyInfo dependency_info = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency_info.bufferMemoryBarrierCount = buffer_barriers.size();
    dependency_info.pBufferMemoryBarriers = buffer_barriers.data();
    vkCmdPipelineBarrier2(command_buffer, &dependency_info);

    std::vector<VkDescriptorSet> descriptors = {
        indirect_descriptors_[frame_index]};
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 1, descriptors.size(),
                            descriptors.data(), 0, nullptr);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      indirect_pipeline_);

    shader::radixsort::PushConstants push_constants;
    push_constants.g_shift = 0;
    push_constants.g_num_blocks_per_workgroup = NUM_BLOCKS_PER_WORKGROUP;
    vkCmdPushConstants(command_buffer, pipeline_layout_,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants),
                       &push_constants);

    vkCmdDispatch(command_buffer, 1, 1, 1);

    buffer_barriers.resize(1);
    buffer_barriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    buffer_barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    buffer_barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    buffer_barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    buffer_barriers[0].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    buffer_barriers[0].buffer = indirect_buffer_;
    buffer_barriers[0].offset = 0;
    buffer_barriers[0].size = indirect_buffer_.size();

    dependency_info = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency_info.bufferMemoryBarrierCount = buffer_barriers.size();
    dependency_info.pBufferMemoryBarriers = buffer_barriers.data();
    vkCmdPipelineBarrier2(command_buffer, &dependency_info);

    // dispatches
    for (int i = 0; i < 4; i++) {
      uint32_t shift = i * 8;

      VkBuffer value_src_buffer = i % 2 == 0 ? values : value_buffer_;
      VkBuffer index_src_buffer = i % 2 == 0 ? indices : index_buffer_;
      VkBuffer value_dst_buffer = i % 2 == 0 ? value_buffer_ : values;
      VkBuffer index_dst_buffer = i % 2 == 0 ? index_buffer_ : indices;

      shader::radixsort::PushConstants push_constants;
      push_constants.g_shift = shift;
      push_constants.g_num_blocks_per_workgroup = NUM_BLOCKS_PER_WORKGROUP;

      // histogram
      std::vector<VkDescriptorSet> descriptors = {
          descriptors_[frame_index * 2 + (i % 2)],
          indirect_descriptors_[frame_index]};
      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipeline_layout_, 0, descriptors.size(),
                              descriptors.data(), 0, nullptr);

      vkCmdPushConstants(command_buffer, pipeline_layout_,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants),
                         &push_constants);

      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        histogram_pipeline_);

      vkCmdDispatchIndirect(command_buffer, indirect_buffer_, 0);

      // barrier
      std::vector<VkBufferMemoryBarrier2> buffer_barriers(1);
      buffer_barriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
      buffer_barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      buffer_barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
      buffer_barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      buffer_barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
      buffer_barriers[0].buffer = histogram_buffer_;
      buffer_barriers[0].offset = 0;
      buffer_barriers[0].size = histogram_buffer_size;

      VkDependencyInfo dependency_info = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      dependency_info.bufferMemoryBarrierCount = buffer_barriers.size();
      dependency_info.pBufferMemoryBarriers = buffer_barriers.data();
      vkCmdPipelineBarrier2(command_buffer, &dependency_info);

      // radix sort
      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        radixsort_pipeline_);

      vkCmdDispatchIndirect(command_buffer, indirect_buffer_, 0);

      // barriers
      if (i < 3) {
        buffer_barriers.resize(5);
        buffer_barriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[0].srcStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        buffer_barriers[0].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        buffer_barriers[0].buffer = value_src_buffer;
        buffer_barriers[0].offset = 0;
        buffer_barriers[0].size = buffer_size;

        buffer_barriers[1] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[1].srcStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[1].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        buffer_barriers[1].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        buffer_barriers[1].buffer = value_dst_buffer;
        buffer_barriers[1].offset = 0;
        buffer_barriers[1].size = buffer_size;

        buffer_barriers[2] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[2].srcStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[2].srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        buffer_barriers[2].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[2].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        buffer_barriers[2].buffer = histogram_buffer_;
        buffer_barriers[2].offset = 0;
        buffer_barriers[2].size = histogram_buffer_size;

        buffer_barriers[3] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[3].srcStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[3].srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        buffer_barriers[3].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[3].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        buffer_barriers[3].buffer = index_src_buffer;
        buffer_barriers[3].offset = 0;
        buffer_barriers[3].size = buffer_size;

        buffer_barriers[4] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[4].srcStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[4].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        buffer_barriers[4].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[4].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        buffer_barriers[4].buffer = index_dst_buffer;
        buffer_barriers[4].offset = 0;
        buffer_barriers[4].size = buffer_size;

        dependency_info = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency_info.bufferMemoryBarrierCount = buffer_barriers.size();
        dependency_info.pBufferMemoryBarriers = buffer_barriers.data();
        vkCmdPipelineBarrier2(command_buffer, &dependency_info);
      }
    }
  }

 private:
  Context context_;
  size_t max_num_elements_ = 0;

  DescriptorLayout descriptor_layout_;
  DescriptorLayout indirect_layout_;
  PipelineLayout pipeline_layout_;
  ComputePipeline indirect_pipeline_;
  ComputePipeline histogram_pipeline_;
  ComputePipeline radixsort_pipeline_;

  std::vector<Descriptor> descriptors_;
  std::vector<Descriptor> indirect_descriptors_;

  static constexpr uint32_t WORKGROUP_SIZE = 256;
  static constexpr uint32_t RADIX_SORT_BINS = 256;
  static constexpr uint32_t NUM_BLOCKS_PER_WORKGROUP = 32;

  Buffer histogram_buffer_;
  Buffer value_buffer_;
  Buffer index_buffer_;
  Buffer indirect_buffer_;
};

Radixsort::Radixsort() = default;

Radixsort::Radixsort(Context context, size_t max_num_elements)
    : impl_(std::make_shared<Impl>(context, max_num_elements)) {}

Radixsort::~Radixsort() = default;

void Radixsort::Sort(VkCommandBuffer command_buffer, uint32_t frame_index,
                     VkBuffer num_elements_buffer, VkBuffer values,
                     VkBuffer indices) {
  impl_->Sort(command_buffer, frame_index, num_elements_buffer, values,
              indices);
}

}  // namespace vk
}  // namespace pygs
