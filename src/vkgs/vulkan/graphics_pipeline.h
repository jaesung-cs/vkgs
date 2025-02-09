#ifndef VKGS_VULKAN_GRAPHICS_PIPELINE_H
#define VKGS_VULKAN_GRAPHICS_PIPELINE_H

#include <memory>
#include <string>

#include <vulkan/vulkan.h>

#include "vkgs/vulkan/context.h"
#include "vkgs/vulkan/pipeline_layout.h"
#include "vkgs/vulkan/shader_module.h"

namespace vkgs {
namespace vk {

struct GraphicsPipelineCreateInfo {
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkRenderPass render_pass = VK_NULL_HANDLE;
  uint32_t subpass = 0;
  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
  ShaderSource vertex_shader;
  ShaderSource geometry_shader;
  ShaderSource fragment_shader;
  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  std::vector<VkVertexInputBindingDescription> input_bindings;
  std::vector<VkVertexInputAttributeDescription> input_attributes;
  bool depth_test = false;
  bool depth_write = false;
  std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments;
};

class GraphicsPipeline {
 public:
  GraphicsPipeline();

  GraphicsPipeline(Context context, const GraphicsPipelineCreateInfo& create_info);

  ~GraphicsPipeline();

  operator VkPipeline() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_VULKAN_GRAPHICS_PIPELINE_H
