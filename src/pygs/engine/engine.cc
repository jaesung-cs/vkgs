#include <pygs/engine/engine.h>

#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include <pygs/scene/camera.h>
#include <pygs/scene/splats.h>

#include "vulkan/context.h"
#include "vulkan/swapchain.h"
#include "vulkan/attachment.h"
#include "vulkan/descriptor_layout.h"
#include "vulkan/pipeline_layout.h"
#include "vulkan/compute_pipeline.h"
#include "vulkan/graphics_pipeline.h"
#include "vulkan/render_pass.h"
#include "vulkan/framebuffer.h"
#include "vulkan/descriptor.h"
#include "vulkan/buffer.h"
#include "vulkan/uniform_buffer.h"
#include "vulkan/radixsort.h"
#include "vulkan/shader/uniforms.h"
#include "vulkan/shader/projection.h"
#include "vulkan/shader/order.h"
#include "vulkan/shader/splat.h"
#include "vulkan/shader/splat_outline.h"

namespace pygs {

class Engine::Impl {
 public:
  Impl() {
    if (glfwInit() == GLFW_FALSE)
      throw std::runtime_error("Failed to initialize glfw.");

    context_ = vk::Context(0);

    // render pass
    render_pass_ = vk::RenderPass(context_, vk::RenderPassType::NORMAL);

    {
      vk::DescriptorLayoutCreateInfo descriptor_layout_info = {};
      descriptor_layout_info.bindings.resize(1);
      descriptor_layout_info.bindings[0] = {};
      descriptor_layout_info.bindings[0].binding = 0;
      descriptor_layout_info.bindings[0].descriptor_type =
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      descriptor_layout_info.bindings[0].stage_flags =
          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
      camera_descriptor_layout_ =
          vk::DescriptorLayout(context_, descriptor_layout_info);
    }

    {
      vk::DescriptorLayoutCreateInfo descriptor_layout_info = {};
      descriptor_layout_info.bindings.resize(8);
      descriptor_layout_info.bindings[0] = {};
      descriptor_layout_info.bindings[0].binding = 0;
      descriptor_layout_info.bindings[0].descriptor_type =
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      descriptor_layout_info.bindings[0].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[1] = {};
      descriptor_layout_info.bindings[1].binding = 1;
      descriptor_layout_info.bindings[1].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[1].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[2] = {};
      descriptor_layout_info.bindings[2].binding = 2;
      descriptor_layout_info.bindings[2].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[2].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[3] = {};
      descriptor_layout_info.bindings[3].binding = 3;
      descriptor_layout_info.bindings[3].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[3].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[4] = {};
      descriptor_layout_info.bindings[4].binding = 4;
      descriptor_layout_info.bindings[4].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[4].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[5] = {};
      descriptor_layout_info.bindings[5].binding = 5;
      descriptor_layout_info.bindings[5].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[5].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[6] = {};
      descriptor_layout_info.bindings[6].binding = 6;
      descriptor_layout_info.bindings[6].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[6].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[7] = {};
      descriptor_layout_info.bindings[7].binding = 7;
      descriptor_layout_info.bindings[7].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[7].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      gaussian_descriptor_layout_ =
          vk::DescriptorLayout(context_, descriptor_layout_info);
    }

    {
      vk::DescriptorLayoutCreateInfo descriptor_layout_info = {};
      descriptor_layout_info.bindings.resize(4);
      descriptor_layout_info.bindings[0] = {};
      descriptor_layout_info.bindings[0].binding = 0;
      descriptor_layout_info.bindings[0].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[0].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[1] = {};
      descriptor_layout_info.bindings[1].binding = 1;
      descriptor_layout_info.bindings[1].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[1].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[2] = {};
      descriptor_layout_info.bindings[2].binding = 2;
      descriptor_layout_info.bindings[2].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[2].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[3] = {};
      descriptor_layout_info.bindings[3].binding = 3;
      descriptor_layout_info.bindings[3].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[3].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      instance_layout_ = vk::DescriptorLayout(context_, descriptor_layout_info);
    }

    // compute pipeline layout
    {
      vk::PipelineLayoutCreateInfo pipeline_layout_info = {};
      pipeline_layout_info.layouts = {camera_descriptor_layout_,
                                      gaussian_descriptor_layout_,
                                      instance_layout_};
      compute_pipeline_layout_ =
          vk::PipelineLayout(context_, pipeline_layout_info);
    }

    // graphics pipeline layout
    {
      vk::PipelineLayoutCreateInfo pipeline_layout_info = {};
      pipeline_layout_info.layouts = {camera_descriptor_layout_};
      graphics_pipeline_layout_ =
          vk::PipelineLayout(context_, pipeline_layout_info);
    }

    // order pipeline
    {
      vk::ComputePipelineCreateInfo pipeline_info = {};
      pipeline_info.layout = compute_pipeline_layout_;
      pipeline_info.compute_shader = vk::shader::order_comp;
      order_pipeline_ = vk::ComputePipeline(context_, pipeline_info);
    }

    // projection pipeline
    {
      vk::ComputePipelineCreateInfo pipeline_info = {};
      pipeline_info.layout = compute_pipeline_layout_;
      pipeline_info.compute_shader = vk::shader::projection_comp;
      projection_pipeline_ = vk::ComputePipeline(context_, pipeline_info);
    }

    // splat pipeline
    {
      std::vector<VkVertexInputBindingDescription> input_bindings(2);
      // xy
      input_bindings[0].binding = 0;
      input_bindings[0].stride = sizeof(float) * 2;
      input_bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

      // ndc position, cov2d, rgba
      input_bindings[1].binding = 1;
      input_bindings[1].stride = sizeof(float) * 10;
      input_bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

      std::vector<VkVertexInputAttributeDescription> input_attributes(4);
      // vertex position
      input_attributes[0].location = 0;
      input_attributes[0].binding = 0;
      input_attributes[0].format = VK_FORMAT_R32G32_SFLOAT;
      input_attributes[0].offset = 0;

      // cov2d
      input_attributes[1].location = 1;
      input_attributes[1].binding = 1;
      input_attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
      input_attributes[1].offset = 0;

      // projected position
      input_attributes[2].location = 2;
      input_attributes[2].binding = 1;
      input_attributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
      input_attributes[2].offset = sizeof(float) * 3;

      // point rgba
      input_attributes[3].location = 3;
      input_attributes[3].binding = 1;
      input_attributes[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
      input_attributes[3].offset = sizeof(float) * 6;

      std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments(
          1);
      color_blend_attachments[0] = {};
      color_blend_attachments[0].blendEnable = VK_TRUE;
      color_blend_attachments[0].srcColorBlendFactor =
          VK_BLEND_FACTOR_SRC_ALPHA;
      color_blend_attachments[0].dstColorBlendFactor =
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachments[0].colorBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      color_blend_attachments[0].dstAlphaBlendFactor =
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachments[0].colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

      vk::GraphicsPipelineCreateInfo pipeline_info = {};
      pipeline_info.layout = graphics_pipeline_layout_;
      pipeline_info.render_pass = render_pass_;
      pipeline_info.vertex_shader = vk::shader::splat_vert;
      pipeline_info.fragment_shader = vk::shader::splat_frag;
      pipeline_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
      pipeline_info.input_bindings = std::move(input_bindings);
      pipeline_info.input_attributes = std::move(input_attributes);
      pipeline_info.depth_test = true;
      pipeline_info.depth_write = false;
      pipeline_info.color_blend_attachments =
          std::move(color_blend_attachments);
      splat_pipeline_ = vk::GraphicsPipeline(context_, pipeline_info);
    }

    // uniforms and descriptors
    camera_buffer_ = vk::UniformBuffer<vk::shader::Camera>(context_, 2);
    camera_descriptors_.resize(2);
    for (int i = 0; i < 2; i++) {
      camera_descriptors_[i] =
          vk::Descriptor(context_, camera_descriptor_layout_);
      camera_descriptors_[i].Update(0, camera_buffer_, camera_buffer_.offset(i),
                                    camera_buffer_.element_size());
    }

    splat_info_buffer_ = vk::UniformBuffer<vk::shader::SplatInfo>(context_, 1);

    splat_indirect_buffer_ = vk::Buffer(
        context_, 5 * sizeof(int),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

    // commands and synchronizations
    draw_command_buffers_.resize(3);
    VkCommandBufferAllocateInfo command_buffer_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_buffer_info.commandPool = context_.command_pool();
    command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_info.commandBufferCount = draw_command_buffers_.size();
    vkAllocateCommandBuffers(context_.device(), &command_buffer_info,
                             draw_command_buffers_.data());

    VkSemaphoreCreateInfo semaphore_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    image_acquired_semaphores_.resize(2);
    render_finished_semaphores_.resize(2);
    render_finished_fences_.resize(2);
    for (int i = 0; i < 2; i++) {
      vkCreateSemaphore(context_.device(), &semaphore_info, NULL,
                        &image_acquired_semaphores_[i]);
      vkCreateSemaphore(context_.device(), &semaphore_info, NULL,
                        &render_finished_semaphores_[i]);
      vkCreateFence(context_.device(), &fence_info, NULL,
                    &render_finished_fences_[i]);
    }

    vkCreateSemaphore(context_.device(), &semaphore_info, NULL,
                      &transfer_semaphore_);

    fence_info.flags = 0;
    vkCreateFence(context_.device(), &fence_info, NULL, &sort_fence_);
  }

  ~Impl() {
    vkDeviceWaitIdle(context_.device());

    for (auto semaphore : image_acquired_semaphores_)
      vkDestroySemaphore(context_.device(), semaphore, NULL);
    for (auto semaphore : render_finished_semaphores_)
      vkDestroySemaphore(context_.device(), semaphore, NULL);
    for (auto fence : render_finished_fences_)
      vkDestroyFence(context_.device(), fence, NULL);
    vkDestroySemaphore(context_.device(), transfer_semaphore_, NULL);

    vkDestroyFence(context_.device(), sort_fence_, NULL);

    glfwTerminate();
  }

  void AddSplats(const Splats& splats) {
    point_count_ = splats.size();
    const auto& position = splats.positions();
    const auto& sh0 = splats.sh0();
    const auto& sh1 = splats.sh1();
    const auto& sh2 = splats.sh2();
    const auto& sh3 = splats.sh3();
    const auto& opacity = splats.opacity();
    const auto& rotation = splats.rots();
    const auto& scale = splats.scales();

    std::vector<float> gaussian_cov3d;
    gaussian_cov3d.reserve(6 * point_count_);
    for (int i = 0; i < point_count_; i++) {
      glm::quat q(rotation[i * 4 + 0], rotation[i * 4 + 1], rotation[i * 4 + 2],
                  rotation[i * 4 + 3]);
      glm::mat3 r = glm::toMat3(q);
      glm::mat3 s = glm::mat4(1.f);
      s[0][0] = scale[i * 3 + 0];
      s[1][1] = scale[i * 3 + 1];
      s[2][2] = scale[i * 3 + 2];
      glm::mat3 m = r * s * s * glm::transpose(r);  // cov = RSSR^T

      gaussian_cov3d.push_back(m[0][0]);
      gaussian_cov3d.push_back(m[1][0]);
      gaussian_cov3d.push_back(m[2][0]);
      gaussian_cov3d.push_back(m[1][1]);
      gaussian_cov3d.push_back(m[2][1]);
      gaussian_cov3d.push_back(m[2][2]);
    }

    std::vector<float> splat_vertex = {
        // xy, ccw in NDC space.
        -1.f, -1.f,  // 0
        -1.f, 1.f,   // 1
        1.f,  -1.f,  // 2
        1.f,  1.f,   // 3
    };
    std::vector<uint32_t> splat_index = {0, 1, 2, 3};

    gaussian_cov3d_buffer_ = vk::Buffer(
        context_, gaussian_cov3d.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    gaussian_position_buffer_ = vk::Buffer(
        context_, position.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    gaussian_opacity_buffer_ = vk::Buffer(
        context_, opacity.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    gaussian_sh0_buffer_ = vk::Buffer(
        context_, sh0.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    gaussian_sh1_buffer_ = vk::Buffer(
        context_, sh1.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    gaussian_sh2_buffer_ = vk::Buffer(
        context_, sh2.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    gaussian_sh3_buffer_ = vk::Buffer(
        context_, sh3.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    splat_vertex_buffer_ = vk::Buffer(
        context_, splat_vertex.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    splat_index_buffer_ = vk::Buffer(
        context_, splat_index.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    splat_instance_buffer_ = vk::Buffer(
        context_, point_count_ * 10 * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    instance_key_buffer_ = vk::Buffer(
        context_, point_count_ * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    instance_index_buffer_ = vk::Buffer(
        context_, point_count_ * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    splat_info_buffer_[0].point_count = point_count_;

    VkCommandBufferAllocateInfo command_buffer_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_buffer_info.commandPool = context_.command_pool();
    command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_info.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(context_.device(), &command_buffer_info, &cb);

    VkCommandBufferBeginInfo begin_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &begin_info);

    gaussian_cov3d_buffer_.FromCpu(cb, gaussian_cov3d);
    gaussian_position_buffer_.FromCpu(cb, position);
    gaussian_opacity_buffer_.FromCpu(cb, opacity);
    gaussian_sh0_buffer_.FromCpu(cb, sh0);
    gaussian_sh1_buffer_.FromCpu(cb, sh1);
    gaussian_sh2_buffer_.FromCpu(cb, sh2);
    gaussian_sh3_buffer_.FromCpu(cb, sh3);
    splat_vertex_buffer_.FromCpu(cb, splat_vertex);
    splat_index_buffer_.FromCpu(cb, splat_index);

    vkEndCommandBuffer(cb);

    std::vector<VkCommandBufferSubmitInfo> command_buffer_submit_info(1);
    command_buffer_submit_info[0] = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    command_buffer_submit_info[0].commandBuffer = cb;

    // TODO: use timeline semaphore to support multiple transfers
    std::vector<VkSemaphoreSubmitInfo> signal_semaphore_info(1);
    signal_semaphore_info[0] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal_semaphore_info[0].semaphore = transfer_semaphore_;
    signal_semaphore_info[0].stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

    VkSubmitInfo2 submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit_info.commandBufferInfoCount = command_buffer_submit_info.size();
    submit_info.pCommandBufferInfos = command_buffer_submit_info.data();
    submit_info.signalSemaphoreInfoCount = signal_semaphore_info.size();
    submit_info.pSignalSemaphoreInfos = signal_semaphore_info.data();
    vkQueueSubmit2(context_.queue(), 1, &submit_info, NULL);

    on_transfer_ = true;

    // update descriptor
    // TODO: make sure descriptors are not in use
    gaussian_descriptor_ =
        vk::Descriptor(context_, gaussian_descriptor_layout_);
    gaussian_descriptor_.Update(0, splat_info_buffer_, 0,
                                splat_info_buffer_.element_size());
    gaussian_descriptor_.Update(1, gaussian_position_buffer_, 0,
                                gaussian_position_buffer_.size());
    gaussian_descriptor_.Update(2, gaussian_cov3d_buffer_, 0,
                                gaussian_cov3d_buffer_.size());
    gaussian_descriptor_.Update(3, gaussian_opacity_buffer_, 0,
                                gaussian_opacity_buffer_.size());
    gaussian_descriptor_.Update(4, gaussian_sh0_buffer_, 0,
                                gaussian_sh0_buffer_.size());
    gaussian_descriptor_.Update(5, gaussian_sh1_buffer_, 0,
                                gaussian_sh1_buffer_.size());
    gaussian_descriptor_.Update(6, gaussian_sh2_buffer_, 0,
                                gaussian_sh2_buffer_.size());
    gaussian_descriptor_.Update(7, gaussian_sh3_buffer_, 0,
                                gaussian_sh3_buffer_.size());

    splat_instance_descriptor_ = vk::Descriptor(context_, instance_layout_);
    splat_instance_descriptor_.Update(0, splat_indirect_buffer_, 0,
                                      splat_indirect_buffer_.size());
    splat_instance_descriptor_.Update(1, splat_instance_buffer_, 0,
                                      splat_instance_buffer_.size());
    splat_instance_descriptor_.Update(2, instance_key_buffer_, 0,
                                      instance_key_buffer_.size());
    splat_instance_descriptor_.Update(3, instance_index_buffer_, 0,
                                      instance_index_buffer_.size());

    // create sorter
    radix_sorter_ = vk::Radixsort(context_, point_count_);
  }

  void Draw(Window window, const Camera& camera) {
    /* projection:
    0.974279 0 0 0
    0 -1.73205 0 0
    0 0 -1.0001 -0.010001
    0 0 -1 0
    */
    auto window_ptr = window.window();
    if (swapchains_.count(window_ptr) == 0) {
      VkSurfaceKHR surface;
      glfwCreateWindowSurface(context_.instance(), window_ptr, NULL, &surface);
      auto swapchain = vk::Swapchain(context_, surface);
      swapchains_[window_ptr] = swapchain;

      color_attachment_ =
          vk::Attachment(context_, swapchain.width(), swapchain.height(),
                         VK_FORMAT_B8G8R8A8_SRGB, VK_SAMPLE_COUNT_4_BIT, false);
      depth_attachment_ = vk::Attachment(
          context_, swapchain.width(), swapchain.height(),
          VK_FORMAT_D24_UNORM_S8_UINT, VK_SAMPLE_COUNT_4_BIT, false);

      vk::FramebufferCreateInfo framebuffer_info;
      framebuffer_info.render_pass = render_pass_;
      framebuffer_info.width = swapchain.width();
      framebuffer_info.height = swapchain.height();
      framebuffer_info.image_specs = {color_attachment_.image_spec(),
                                      depth_attachment_.image_spec(),
                                      swapchain.image_spec()};
      framebuffer_ = vk::Framebuffer(context_, framebuffer_info);
    }

    auto swapchain = swapchains_[window_ptr];

    if (swapchain.ShouldRecreate()) {
      vkWaitForFences(context_.device(), render_finished_fences_.size(),
                      render_finished_fences_.data(), VK_TRUE, UINT64_MAX);
      swapchain.Recreate();

      color_attachment_ =
          vk::Attachment(context_, swapchain.width(), swapchain.height(),
                         VK_FORMAT_B8G8R8A8_SRGB, VK_SAMPLE_COUNT_4_BIT, false);
      depth_attachment_ = vk::Attachment(
          context_, swapchain.width(), swapchain.height(),
          VK_FORMAT_D24_UNORM_S8_UINT, VK_SAMPLE_COUNT_4_BIT, false);

      vk::FramebufferCreateInfo framebuffer_info;
      framebuffer_info.render_pass = render_pass_;
      framebuffer_info.width = swapchain.width();
      framebuffer_info.height = swapchain.height();
      framebuffer_info.image_specs = {color_attachment_.image_spec(),
                                      depth_attachment_.image_spec(),
                                      swapchain.image_spec()};
      framebuffer_ = vk::Framebuffer(context_, framebuffer_info);
    }

    int32_t frame_index = frame_counter_ % 2;
    VkSemaphore image_acquired_semaphore =
        image_acquired_semaphores_[frame_index];
    VkSemaphore render_finished_semaphore =
        render_finished_semaphores_[frame_index];
    VkFence render_finished_fence = render_finished_fences_[frame_index];
    VkCommandBuffer cb = draw_command_buffers_[frame_index];

    uint32_t image_index;
    if (swapchain.AcquireNextImage(image_acquired_semaphore, &image_index)) {
      vkWaitForFences(context_.device(), 1, &render_finished_fence, VK_TRUE,
                      UINT64_MAX);
      vkResetFences(context_.device(), 1, &render_finished_fence);

      camera_buffer_[frame_index].projection = camera.ProjectionMatrix();
      camera_buffer_[frame_index].view = camera.ViewMatrix();
      camera_buffer_[frame_index].camera_position = camera.Eye();

      VkCommandBufferBeginInfo command_begin_info = {
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      command_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(cb, &command_begin_info);

      // order
      {
        std::vector<VkBufferMemoryBarrier2> buffer_barriers(1);
        buffer_barriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        buffer_barriers[0].srcAccessMask =
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        buffer_barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        buffer_barriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        buffer_barriers[0].buffer = splat_indirect_buffer_;
        buffer_barriers[0].offset = 0;
        buffer_barriers[0].size = splat_indirect_buffer_.size();

        VkDependencyInfo barrier = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        barrier.bufferMemoryBarrierCount = buffer_barriers.size();
        barrier.pBufferMemoryBarriers = buffer_barriers.data();
        vkCmdPipelineBarrier2(cb, &barrier);

        VkDrawIndexedIndirectCommand draw_indexed_indirect = {};
        draw_indexed_indirect.indexCount = 4;  // quad
        draw_indexed_indirect.instanceCount = 0;
        draw_indexed_indirect.firstIndex = 0;
        draw_indexed_indirect.vertexOffset = 0;
        draw_indexed_indirect.firstInstance = 0;
        vkCmdUpdateBuffer(cb, splat_indirect_buffer_, 0,
                          sizeof(draw_indexed_indirect),
                          &draw_indexed_indirect);

        buffer_barriers.resize(3);
        buffer_barriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        buffer_barriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        buffer_barriers[0].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        buffer_barriers[0].buffer = splat_indirect_buffer_;
        buffer_barriers[0].offset = 0;
        buffer_barriers[0].size = splat_indirect_buffer_.size();

        buffer_barriers[1] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[1].srcStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[1].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        buffer_barriers[1].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        buffer_barriers[1].buffer = instance_key_buffer_;
        buffer_barriers[1].offset = 0;
        buffer_barriers[1].size = instance_key_buffer_.size();

        buffer_barriers[2] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[2].srcStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[2].srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        buffer_barriers[2].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[2].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        buffer_barriers[2].buffer = instance_index_buffer_;
        buffer_barriers[2].offset = 0;
        buffer_barriers[2].size = instance_index_buffer_.size();

        barrier = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        barrier.bufferMemoryBarrierCount = buffer_barriers.size();
        barrier.pBufferMemoryBarriers = buffer_barriers.data();
        vkCmdPipelineBarrier2(cb, &barrier);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, order_pipeline_);

        std::vector<VkDescriptorSet> descriptors = {
            camera_descriptors_[frame_index],
            gaussian_descriptor_,
            splat_instance_descriptor_,
        };
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compute_pipeline_layout_, 0, descriptors.size(),
                                descriptors.data(), 0, nullptr);

        constexpr int local_size = 256;
        vkCmdDispatch(cb, (point_count_ + local_size - 1) / local_size, 1, 1);
      }

      // radix sort
      {
        std::vector<VkBufferMemoryBarrier2> buffer_barriers(3);
        buffer_barriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[0].srcStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        buffer_barriers[0].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        buffer_barriers[0].buffer = splat_indirect_buffer_;
        buffer_barriers[0].offset = 1 * sizeof(uint32_t);
        buffer_barriers[0].size = sizeof(uint32_t);

        buffer_barriers[1] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[1].srcStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[1].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        buffer_barriers[1].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        buffer_barriers[1].buffer = instance_key_buffer_;
        buffer_barriers[1].offset = 0;
        buffer_barriers[1].size = instance_key_buffer_.size();

        buffer_barriers[2] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[2].srcStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[2].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        buffer_barriers[2].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[2].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        buffer_barriers[2].buffer = instance_index_buffer_;
        buffer_barriers[2].offset = 0;
        buffer_barriers[2].size = instance_index_buffer_.size();

        VkDependencyInfo barrier = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        barrier.bufferMemoryBarrierCount = buffer_barriers.size();
        barrier.pBufferMemoryBarriers = buffer_barriers.data();
        vkCmdPipelineBarrier2(cb, &barrier);

        radix_sorter_.Sort(cb, frame_index, splat_indirect_buffer_,
                           instance_key_buffer_, instance_index_buffer_);
      }

      // projection
      {
        std::vector<VkBufferMemoryBarrier2> buffer_barriers(3);
        buffer_barriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        buffer_barriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        buffer_barriers[0].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        buffer_barriers[0].buffer = splat_indirect_buffer_;
        buffer_barriers[0].offset = 0;
        buffer_barriers[0].size = splat_indirect_buffer_.size();

        buffer_barriers[1] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        buffer_barriers[1].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        buffer_barriers[1].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        buffer_barriers[1].buffer = instance_index_buffer_;
        buffer_barriers[1].offset = 0;
        buffer_barriers[1].size = instance_index_buffer_.size();

        buffer_barriers[2] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[2].srcStageMask =
            VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
        buffer_barriers[2].srcAccessMask =
            VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        buffer_barriers[2].dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[2].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        buffer_barriers[2].buffer = splat_instance_buffer_;
        buffer_barriers[2].offset = 0;
        buffer_barriers[2].size = splat_instance_buffer_.size();

        VkDependencyInfo barrier = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        barrier.bufferMemoryBarrierCount = buffer_barriers.size();
        barrier.pBufferMemoryBarriers = buffer_barriers.data();
        vkCmdPipelineBarrier2(cb, &barrier);

        std::vector<VkDescriptorSet> descriptors = {
            camera_descriptors_[frame_index], gaussian_descriptor_,
            splat_instance_descriptor_};
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compute_pipeline_layout_, 0, descriptors.size(),
                                descriptors.data(), 0, nullptr);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          projection_pipeline_);

        constexpr int local_size = 256;
        vkCmdDispatch(cb, (point_count_ + local_size - 1) / local_size, 1, 1);
      }

      // draw
      {
        std::vector<VkBufferMemoryBarrier2> buffer_barriers(2);
        buffer_barriers[0] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        buffer_barriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        buffer_barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        buffer_barriers[0].dstAccessMask =
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        buffer_barriers[0].buffer = splat_indirect_buffer_;
        buffer_barriers[0].offset = 0;
        buffer_barriers[0].size = splat_indirect_buffer_.size();

        buffer_barriers[1] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        buffer_barriers[1].srcStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        buffer_barriers[1].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        buffer_barriers[1].dstStageMask =
            VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
        buffer_barriers[1].dstAccessMask =
            VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        buffer_barriers[1].buffer = splat_instance_buffer_;
        buffer_barriers[1].offset = 0;
        buffer_barriers[1].size = splat_instance_buffer_.size();

        VkDependencyInfo barrier = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        barrier.bufferMemoryBarrierCount = buffer_barriers.size();
        barrier.pBufferMemoryBarriers = buffer_barriers.data();
        vkCmdPipelineBarrier2(cb, &barrier);

        DrawNormalPass(cb, frame_index, swapchain.width(), swapchain.height(),
                       swapchain.image_view(image_index));
      }

      vkEndCommandBuffer(cb);

      std::vector<VkSemaphoreSubmitInfo> wait_semaphores(1);
      wait_semaphores[0] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      wait_semaphores[0].semaphore = image_acquired_semaphore;
      wait_semaphores[0].stageMask = 0;

      if (on_transfer_) {
        VkSemaphoreSubmitInfo wait_semaphore = {
            VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        wait_semaphore.semaphore = transfer_semaphore_;
        wait_semaphore.stageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
        wait_semaphores.push_back(wait_semaphore);
        on_transfer_ = false;
      }

      std::vector<VkCommandBufferSubmitInfo> command_buffer_submit_info(1);
      command_buffer_submit_info[0] = {
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
      command_buffer_submit_info[0].commandBuffer = cb;

      std::vector<VkSemaphoreSubmitInfo> signal_semaphores(1);
      signal_semaphores[0] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      signal_semaphores[0].semaphore = render_finished_semaphore;
      signal_semaphores[0].stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

      VkSubmitInfo2 submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submit_info.waitSemaphoreInfoCount = wait_semaphores.size();
      submit_info.pWaitSemaphoreInfos = wait_semaphores.data();
      submit_info.commandBufferInfoCount = command_buffer_submit_info.size();
      submit_info.pCommandBufferInfos = command_buffer_submit_info.data();
      submit_info.signalSemaphoreInfoCount = signal_semaphores.size();
      submit_info.pSignalSemaphoreInfos = signal_semaphores.data();
      vkQueueSubmit2(context_.queue(), 1, &submit_info, render_finished_fence);

      VkSwapchainKHR swapchain_handle = swapchain.swapchain();
      VkPresentInfoKHR present_info = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
      present_info.waitSemaphoreCount = 1;
      present_info.pWaitSemaphores = &render_finished_semaphore;
      present_info.swapchainCount = 1;
      present_info.pSwapchains = &swapchain_handle;
      present_info.pImageIndices = &image_index;
      vkQueuePresentKHR(context_.queue(), &present_info);

      frame_counter_++;
    }
  }

 private:
  void DrawNormalPass(VkCommandBuffer cb, uint32_t frame_index, uint32_t width,
                      uint32_t height, VkImageView target_image_view) {
    std::vector<VkClearValue> clear_values(2);
    clear_values[0].color.float32[0] = 0.5f;
    clear_values[0].color.float32[1] = 0.5f;
    clear_values[0].color.float32[2] = 0.5f;
    clear_values[0].color.float32[3] = 1.f;
    clear_values[1].depthStencil.depth = 1.f;

    std::vector<VkImageView> render_pass_attachments = {
        color_attachment_,
        depth_attachment_,
        target_image_view,
    };
    VkRenderPassAttachmentBeginInfo render_pass_attachments_info = {
        VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO};
    render_pass_attachments_info.attachmentCount =
        render_pass_attachments.size();
    render_pass_attachments_info.pAttachments = render_pass_attachments.data();

    VkRenderPassBeginInfo render_pass_begin_info = {
        VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    render_pass_begin_info.pNext = &render_pass_attachments_info;
    render_pass_begin_info.renderPass = render_pass_;
    render_pass_begin_info.framebuffer = framebuffer_;
    render_pass_begin_info.renderArea.offset = {0, 0};
    render_pass_begin_info.renderArea.extent = {width, height};
    render_pass_begin_info.clearValueCount = clear_values.size();
    render_pass_begin_info.pClearValues = clear_values.data();
    vkCmdBeginRenderPass(cb, &render_pass_begin_info,
                         VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = {};
    viewport.x = 0.f;
    viewport.y = 0.f;
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;
    vkCmdSetViewport(cb, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {width, height};
    vkCmdSetScissor(cb, 0, 1, &scissor);

    std::vector<VkDescriptorSet> descriptors = {
        camera_descriptors_[frame_index]};
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            graphics_pipeline_layout_, 0, descriptors.size(),
                            descriptors.data(), 0, nullptr);

    // draw splat
    {
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, splat_pipeline_);

      std::vector<VkBuffer> vbs = {splat_vertex_buffer_,
                                   splat_instance_buffer_};
      std::vector<VkDeviceSize> vb_offsets = {0, 0, 0};
      vkCmdBindVertexBuffers(cb, 0, vbs.size(), vbs.data(), vb_offsets.data());

      vkCmdBindIndexBuffer(cb, splat_index_buffer_, 0, VK_INDEX_TYPE_UINT32);

      vkCmdDrawIndexedIndirect(cb, splat_indirect_buffer_, 0, 1, 0);
    }

    vkCmdEndRenderPass(cb);
  }

  vk::Context context_;
  std::unordered_map<GLFWwindow*, vk::Swapchain> swapchains_;

  std::vector<VkCommandBuffer> draw_command_buffers_;
  std::vector<VkSemaphore> image_acquired_semaphores_;
  std::vector<VkSemaphore> render_finished_semaphores_;
  std::vector<VkFence> render_finished_fences_;

  vk::DescriptorLayout camera_descriptor_layout_;
  vk::DescriptorLayout gaussian_descriptor_layout_;
  vk::DescriptorLayout instance_layout_;
  vk::PipelineLayout compute_pipeline_layout_;
  vk::PipelineLayout graphics_pipeline_layout_;

  // preprocess
  vk::ComputePipeline order_pipeline_;
  vk::ComputePipeline projection_pipeline_;
  vk::Radixsort radix_sorter_;

  // normal pass
  vk::Framebuffer framebuffer_;
  vk::RenderPass render_pass_;
  vk::GraphicsPipeline splat_pipeline_;

  vk::Attachment color_attachment_;
  vk::Attachment depth_attachment_;

  std::vector<vk::Descriptor> camera_descriptors_;
  vk::Descriptor gaussian_descriptor_;
  vk::Descriptor splat_instance_descriptor_;

  vk::UniformBuffer<vk::shader::Camera> camera_buffer_;
  vk::UniformBuffer<vk::shader::SplatInfo> splat_info_buffer_;
  vk::Buffer gaussian_position_buffer_;
  vk::Buffer gaussian_cov3d_buffer_;
  vk::Buffer gaussian_opacity_buffer_;
  vk::Buffer gaussian_sh0_buffer_;
  vk::Buffer gaussian_sh1_buffer_;
  vk::Buffer gaussian_sh2_buffer_;
  vk::Buffer gaussian_sh3_buffer_;

  vk::Buffer instance_key_buffer_;
  vk::Buffer instance_index_buffer_;

  vk::Buffer splat_vertex_buffer_;    // gaussian2d quad
  vk::Buffer splat_index_buffer_;     // gaussian2d quad
  vk::Buffer splat_indirect_buffer_;  // indirect command
  vk::Buffer splat_instance_buffer_;  // output of compute shader

  VkFence sort_fence_ = VK_NULL_HANDLE;

  int point_count_ = 0;
  bool on_transfer_ = false;
  VkSemaphore transfer_semaphore_ = VK_NULL_HANDLE;

  uint64_t frame_counter_ = 0;
};

Engine::Engine() : impl_(std::make_shared<Impl>()) {}

Engine::~Engine() {}

void Engine::AddSplats(const Splats& splats) { impl_->AddSplats(splats); }

void Engine::Draw(Window window, const Camera& camera) {
  impl_->Draw(window, camera);
}

}  // namespace pygs
