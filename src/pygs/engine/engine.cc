#include <pygs/engine/engine.h>

#include <iostream>
#include <unordered_map>

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include <pygs/scene/camera.h>
#include <pygs/scene/splats.h>

#include "vulkan/context.h"
#include "vulkan/swapchain.h"
#include "vulkan/descriptor_layout.h"
#include "vulkan/pipeline_layout.h"
#include "vulkan/graphics_pipeline.h"
#include "vulkan/descriptor.h"
#include "vulkan/buffer.h"
#include "vulkan/uniform_buffer.h"
#include "vulkan/shader/uniforms.h"
#include "vulkan/shader/point.h"
#include "vulkan/shader/axes.h"

namespace pygs {

class Engine::Impl {
 public:
  Impl() {
    if (glfwInit() == GLFW_FALSE)
      throw std::runtime_error("Failed to initialize glfw.");

    context_ = vk::Context(0);

    vk::DescriptorLayoutCreateInfo camera_layout_info = {};
    camera_layout_info.bindings.resize(1);
    camera_layout_info.bindings[0] = {};
    camera_layout_info.bindings[0].binding = 0;
    camera_layout_info.bindings[0].descriptor_type =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    camera_layout_info.bindings[0].stage_flags = VK_SHADER_STAGE_VERTEX_BIT;
    camera_descriptor_layout_ =
        vk::DescriptorLayout(context_, camera_layout_info);

    camera_buffer_ = vk::UniformBuffer<vk::shader::Camera>(context_, 2);
    camera_descriptors_.resize(2);
    for (int i = 0; i < 2; i++) {
      camera_descriptors_[i] =
          vk::Descriptor(context_, camera_descriptor_layout_);
      camera_descriptors_[i].Update(0, camera_buffer_, camera_buffer_.offset(i),
                                    camera_buffer_.element_size());
    }

    vk::PipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.layouts = {camera_descriptor_layout_};
    pipeline_layout_ = vk::PipelineLayout(context_, pipeline_layout_info);

    // point pipeline
    {
      std::vector<VkVertexInputBindingDescription> input_bindings(2);
      input_bindings[0].binding = 0;
      input_bindings[0].stride = sizeof(float) * 3;
      input_bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

      input_bindings[1].binding = 1;
      input_bindings[1].stride = sizeof(float) * 4;
      input_bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

      std::vector<VkVertexInputAttributeDescription> input_attributes(2);
      input_attributes[0].location = 0;
      input_attributes[0].binding = 0;
      input_attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
      input_attributes[0].offset = 0;

      input_attributes[1].location = 1;
      input_attributes[1].binding = 1;
      input_attributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
      input_attributes[1].offset = 0;

      vk::GraphicsPipelineCreateInfo pipeline_info = {};
      pipeline_info.layout = pipeline_layout_;
      pipeline_info.vertex_shader = vk::shader::point_vert;
      pipeline_info.fragment_shader = vk::shader::point_frag;
      pipeline_info.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
      pipeline_info.input_bindings = std::move(input_bindings);
      pipeline_info.input_attributes = std::move(input_attributes);
      point_pipeline_ = vk::GraphicsPipeline(context_, pipeline_info);
    }

    // axes pipeline
    {
      std::vector<VkVertexInputBindingDescription> input_bindings(2);
      // xyzrgb
      input_bindings[0].binding = 0;
      input_bindings[0].stride = sizeof(float) * 6;
      input_bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

      input_bindings[1].binding = 1;
      input_bindings[1].stride = sizeof(float) * 16;
      input_bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

      std::vector<VkVertexInputAttributeDescription> input_attributes(6);
      // vertex position
      input_attributes[0].location = 0;
      input_attributes[0].binding = 0;
      input_attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
      input_attributes[0].offset = 0;

      // vertex color
      input_attributes[1].location = 1;
      input_attributes[1].binding = 0;
      input_attributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
      input_attributes[1].offset = sizeof(float) * 3;

      // instance mat4
      input_attributes[2].location = 2;
      input_attributes[2].binding = 1;
      input_attributes[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
      input_attributes[2].offset = 0;
      input_attributes[3].location = 3;
      input_attributes[3].binding = 1;
      input_attributes[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
      input_attributes[3].offset = sizeof(float) * 4;
      input_attributes[4].location = 4;
      input_attributes[4].binding = 1;
      input_attributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
      input_attributes[4].offset = sizeof(float) * 8;
      input_attributes[5].location = 5;
      input_attributes[5].binding = 1;
      input_attributes[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
      input_attributes[5].offset = sizeof(float) * 12;

      vk::GraphicsPipelineCreateInfo pipeline_info = {};
      pipeline_info.layout = pipeline_layout_;
      pipeline_info.vertex_shader = vk::shader::axes_vert;
      pipeline_info.fragment_shader = vk::shader::axes_frag;
      pipeline_info.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
      pipeline_info.input_bindings = std::move(input_bindings);
      pipeline_info.input_attributes = std::move(input_attributes);
      axes_pipeline_ = vk::GraphicsPipeline(context_, pipeline_info);
    }

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

    glfwTerminate();
  }

  void AddSplats(const Splats& splats) {
    point_count_ = splats.size();
    const auto& position = splats.positions();
    const auto& color = splats.colors();
    const auto& rotation = splats.rots();
    const auto& scale = splats.scales();

    std::vector<float> transform;
    transform.reserve(16 * point_count_);
    for (int i = 0; i < point_count_; i++) {
      glm::quat q(rotation[i * 4 + 0], rotation[i * 4 + 1], rotation[i * 4 + 2],
                  rotation[i * 4 + 3]);
      glm::mat4 r = glm::toMat4(q);
      glm::mat4 s = glm::mat4(1.f);
      s[0][0] = scale[i * 3 + 0];
      s[1][1] = scale[i * 3 + 1];
      s[2][2] = scale[i * 3 + 2];
      glm::mat4 m = r * s;
      m[3][0] = position[i * 3 + 0];
      m[3][1] = position[i * 3 + 1];
      m[3][2] = position[i * 3 + 2];

      for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) transform.push_back(m[c][r]);
      }
    }

    std::cout << transform.size() / 16 << std::endl;

    std::vector<float> axes = {
        -1.f, 0.f,  0.f,  1.f, 0.f, 0.f,  // 0
        1.f,  0.f,  0.f,  1.f, 0.f, 0.f,  // 1
        0.f,  -1.f, 0.f,  0.f, 1.f, 0.f,  // 2
        0.f,  1.f,  0.f,  0.f, 1.f, 0.f,  // 3
        0.f,  0.f,  -1.f, 0.f, 0.f, 1.f,  // 4
        0.f,  0.f,  1.f,  0.f, 0.f, 1.f,  // 5
    };

    position_buffer_ = vk::Buffer(
        context_, position.size() * sizeof(float),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    color_buffer_ = vk::Buffer(
        context_, color.size() * sizeof(float),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    transform_buffer_ = vk::Buffer(
        context_, transform.size() * sizeof(float),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    axes_buffer_ = vk::Buffer(
        context_, axes.size() * sizeof(float),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

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

    position_buffer_.FromCpu(cb, position);
    color_buffer_.FromCpu(cb, color);
    transform_buffer_.FromCpu(cb, transform);
    axes_buffer_.FromCpu(cb, axes);

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
  }

  void Draw(Window window, const Camera& camera) {
    auto window_ptr = window.window();
    if (swapchains_.count(window_ptr) == 0) {
      VkSurfaceKHR surface;
      glfwCreateWindowSurface(context_.instance(), window_ptr, NULL, &surface);
      swapchains_[window_ptr] = vk::Swapchain(context_, surface);
    }

    auto swapchain = swapchains_[window_ptr];

    if (swapchain.ShouldRecreate()) {
      vkWaitForFences(context_.device(), render_finished_fences_.size(),
                      render_finished_fences_.data(), VK_TRUE, UINT64_MAX);
      swapchain.Recreate();
    }

    int32_t frame_index = frame_counter_ % 2;
    VkSemaphore image_acquired_semaphore =
        image_acquired_semaphores_[frame_index];

    VkSemaphore render_finished_semaphore =
        render_finished_semaphores_[frame_index];
    VkFence render_finished_fence = render_finished_fences_[frame_index];
    VkCommandBuffer cb = draw_command_buffers_[frame_index];
    vkWaitForFences(context_.device(), 1, &render_finished_fence, VK_TRUE,
                    UINT64_MAX);
    vkResetFences(context_.device(), 1, &render_finished_fence);

    uint32_t image_index;
    if (swapchain.AcquireNextImage(image_acquired_semaphore, &image_index)) {
      camera_buffer_[frame_index].projection = camera.ProjectionMatrix();
      camera_buffer_[frame_index].view = camera.ViewMatrix();

      VkCommandBufferBeginInfo command_begin_info = {
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      command_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(cb, &command_begin_info);

      // Layout transition
      std::vector<VkImageMemoryBarrier2> image_barriers(1);
      image_barriers[0] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
      image_barriers[0].srcStageMask = 0;
      image_barriers[0].srcAccessMask = 0;
      image_barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
      image_barriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      image_barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      image_barriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      image_barriers[0].image = swapchain.image(image_index);
      image_barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                            1};

      VkDependencyInfo barrier = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      barrier.imageMemoryBarrierCount = image_barriers.size();
      barrier.pImageMemoryBarriers = image_barriers.data();
      vkCmdPipelineBarrier2(cb, &barrier);

      std::vector<VkRenderingAttachmentInfo> color_attachments(1);
      color_attachments[0] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
      color_attachments[0].imageView = swapchain.image_view(image_index);
      color_attachments[0].imageLayout =
          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      color_attachments[0].resolveMode = VK_RESOLVE_MODE_NONE;
      color_attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      color_attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      color_attachments[0].clearValue.color.float32[0] = 0.5f;
      color_attachments[0].clearValue.color.float32[1] = 0.5f;
      color_attachments[0].clearValue.color.float32[2] = 0.5f;
      color_attachments[0].clearValue.color.float32[3] = 1.f;

      VkRenderingInfo rendering_info = {VK_STRUCTURE_TYPE_RENDERING_INFO};
      rendering_info.renderArea.offset = {0, 0};
      rendering_info.renderArea.extent = {swapchain.width(),
                                          swapchain.height()};
      rendering_info.layerCount = 1;
      rendering_info.colorAttachmentCount = color_attachments.size();
      rendering_info.pColorAttachments = color_attachments.data();
      vkCmdBeginRendering(cb, &rendering_info);

      VkViewport viewport = {};
      viewport.x = 0.f;
      viewport.y = 0.f;
      viewport.width = static_cast<float>(swapchain.width());
      viewport.height = static_cast<float>(swapchain.height());
      viewport.minDepth = 0.f;
      viewport.maxDepth = 1.f;
      vkCmdSetViewport(cb, 0, 1, &viewport);

      VkRect2D scissor = {};
      scissor.offset = {0, 0};
      scissor.extent = {swapchain.width(), swapchain.height()};
      vkCmdSetScissor(cb, 0, 1, &scissor);

      std::vector<VkDescriptorSet> descriptors = {
          camera_descriptors_[frame_index]};
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipeline_layout_, 0, descriptors.size(),
                              descriptors.data(), 0, nullptr);

      // draw points
      {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, point_pipeline_);

        std::vector<VkBuffer> vbs = {position_buffer_, color_buffer_};
        std::vector<VkDeviceSize> vb_offsets = {0, 0};
        vkCmdBindVertexBuffers(cb, 0, vbs.size(), vbs.data(),
                               vb_offsets.data());

        vkCmdDraw(cb, point_count_, 1, 0, 0);
      }

      // draw axes
      {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, axes_pipeline_);

        std::vector<VkBuffer> vbs = {axes_buffer_, transform_buffer_};
        std::vector<VkDeviceSize> vb_offsets = {0, 0};
        vkCmdBindVertexBuffers(cb, 0, vbs.size(), vbs.data(),
                               vb_offsets.data());

        vkCmdDraw(cb, 6, point_count_, 0, 0);
      }

      vkCmdEndRendering(cb);

      // Layout transition
      image_barriers[0] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
      image_barriers[0].srcStageMask =
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      image_barriers[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
      image_barriers[0].dstStageMask = 0;
      image_barriers[0].dstAccessMask = 0;
      image_barriers[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      image_barriers[0].newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      image_barriers[0].image = swapchain.image(image_index);
      image_barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                            1};

      barrier = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      barrier.imageMemoryBarrierCount = image_barriers.size();
      barrier.pImageMemoryBarriers = image_barriers.data();
      vkCmdPipelineBarrier2(cb, &barrier);

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
  vk::Context context_;
  std::unordered_map<GLFWwindow*, vk::Swapchain> swapchains_;

  std::vector<VkCommandBuffer> draw_command_buffers_;
  std::vector<VkSemaphore> image_acquired_semaphores_;
  std::vector<VkSemaphore> render_finished_semaphores_;
  std::vector<VkFence> render_finished_fences_;

  vk::DescriptorLayout camera_descriptor_layout_;
  vk::PipelineLayout pipeline_layout_;
  vk::GraphicsPipeline point_pipeline_;
  vk::GraphicsPipeline axes_pipeline_;

  std::vector<vk::Descriptor> camera_descriptors_;
  vk::UniformBuffer<vk::shader::Camera> camera_buffer_;
  vk::Buffer position_buffer_;
  vk::Buffer color_buffer_;
  vk::Buffer transform_buffer_;
  vk::Buffer axes_buffer_;
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
