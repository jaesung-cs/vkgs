#include <pygs/engine/engine.h>

#include <iostream>
#include <unordered_map>

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "vulkan/context.h"
#include "vulkan/swapchain.h"

namespace pygs {

class Engine::Impl {
 public:
  Impl() {
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
    for (int i = 0; i < 2; i++) {
      vkCreateSemaphore(context_.device(), &semaphore_info, NULL,
                        &image_acquired_semaphores_[i]);
    }
    render_finished_semaphores_.resize(3);
    render_finished_fences_.resize(3);
    for (int i = 0; i < 3; i++) {
      vkCreateSemaphore(context_.device(), &semaphore_info, NULL,
                        &render_finished_semaphores_[i]);
      vkCreateFence(context_.device(), &fence_info, NULL,
                    &render_finished_fences_[i]);
    }
  }

  ~Impl() {
    vkDeviceWaitIdle(context_.device());

    for (auto semaphore : image_acquired_semaphores_)
      vkDestroySemaphore(context_.device(), semaphore, NULL);
    for (auto semaphore : render_finished_semaphores_)
      vkDestroySemaphore(context_.device(), semaphore, NULL);
    for (auto fence : render_finished_fences_)
      vkDestroyFence(context_.device(), fence, NULL);
  }

  void Draw(Window window) {
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

    uint32_t image_index;
    if (swapchain.AcquireNextImage(image_acquired_semaphore, &image_index)) {
      VkSemaphore render_finished_semaphore =
          render_finished_semaphores_[image_index];
      VkFence render_finished_fence = render_finished_fences_[image_index];
      VkCommandBuffer cb = draw_command_buffers_[image_index];

      vkWaitForFences(context_.device(), 1, &render_finished_fence, VK_TRUE,
                      UINT64_MAX);
      vkResetFences(context_.device(), 1, &render_finished_fence);

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

      // TODO: draw

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

      std::cout << "frame " << frame_counter_ << std::endl;
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

  uint64_t frame_counter_ = 0;
};

Engine::Engine() : impl_(std::make_shared<Impl>()) {}

Engine::~Engine() {}

void Engine::Draw(Window window) { impl_->Draw(window); }

}  // namespace pygs
