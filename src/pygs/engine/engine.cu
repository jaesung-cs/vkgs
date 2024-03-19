#include <pygs/engine/engine.h>

#include <iostream>
#include <unordered_map>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "vulkan/context.h"
#include "vulkan/cuda_image.h"
#include "vulkan/cuda_semaphore.h"
#include "vulkan/swapchain.h"

namespace pygs {
namespace {

__global__ void test_kernel(int width, int height, float4* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= height || j >= width) return;

  float r = (float)threadIdx.x / blockDim.x;
  float g = (float)threadIdx.y / blockDim.y;
  float b = 0.f;

  out[i * width + j] = {r, g, b, 1.f};
}

}  // namespace

class Engine::Impl {
 public:
  Impl() {
    cuda_semaphore_ = vk::CudaSemaphore(context_);

    VkCommandBufferAllocateInfo command_buffer_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_buffer_info.commandPool = context_.command_pool();
    command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_info.commandBufferCount = 1;
    vkAllocateCommandBuffers(context_.device(), &command_buffer_info, &cb_);

    VkSemaphoreCreateInfo semaphore_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(context_.device(), &semaphore_info, NULL,
                      &image_acquired_semaphore_);
    vkCreateSemaphore(context_.device(), &semaphore_info, NULL,
                      &render_finished_semaphore_);

    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(context_.device(), &fence_info, NULL, &fence_);

    cudaStreamCreate(&stream_);

    cudaMalloc(&test_cuda_mem_, 1600 * 900 * 4 * sizeof(float));
  }

  ~Impl() {
    vkDeviceWaitIdle(context_.device());
    vkDestroySemaphore(context_.device(), image_acquired_semaphore_, NULL);
    vkDestroySemaphore(context_.device(), render_finished_semaphore_, NULL);
    vkDestroyFence(context_.device(), fence_, NULL);
  }

  void Draw(Window window) {
    auto window_ptr = window.window();
    if (swapchains_.count(window_ptr) == 0) {
      swapchains_[window_ptr] = vk::Swapchain(context_, window_ptr);
    }

    vkWaitForFences(context_.device(), 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(context_.device(), 1, &fence_);

    auto swapchain = swapchains_[window_ptr];
    uint32_t image_index =
        swapchain.AcquireNextImage(image_acquired_semaphore_);

    if (!cuda_image_ || cuda_image_.width() != swapchain.width() ||
        cuda_image_.height() != swapchain.height()) {
      cuda_image_ =
          vk::CudaImage(context_, swapchain.width(), swapchain.height());
    }

    dim3 threads(16, 16, 1);
    dim3 blocks((cuda_image_.height() + 15) / 16,
                (cuda_image_.width() + 15) / 16, 1);

    std::cout << (cuda_image_.height() + 15) / 16 << ' '
              << (cuda_image_.width() + 15) / 16 << std::endl;

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    test_kernel<<<blocks, threads, 0, stream_>>>(
        cuda_image_.width(), cuda_image_.height(),
        static_cast<float4*>(cuda_image_.map()));

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed time 1: " << elapsedTime << " ms" << std::endl;

    /*
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    test_kernel<<<blocks, threads, 0, stream_>>>(
        cuda_image_.width(), cuda_image_.height(),
        static_cast<float4*>(test_cuda_mem_));

    cudaMemcpyAsync(cuda_image_.map(), test_cuda_mem_,
                    cuda_image_.width() * cuda_image_.height() * sizeof(float4),
                    cudaMemcpyDefault, stream_);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed time 2: " << elapsedTime << " ms" << std::endl;
    */

    cuda_semaphore_.signal(stream_);

    VkCommandBufferBeginInfo command_begin_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    command_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb_, &command_begin_info);

    // Layout transition
    std::vector<VkImageMemoryBarrier2> image_barriers(2);
    image_barriers[0] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    image_barriers[0].srcStageMask = 0;
    image_barriers[0].srcAccessMask = 0;
    image_barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    image_barriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    image_barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    image_barriers[0].image = cuda_image_.image();
    image_barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                          1};

    image_barriers[1] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    image_barriers[1].srcStageMask = 0;
    image_barriers[1].srcAccessMask = 0;
    image_barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    image_barriers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    image_barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barriers[1].image = swapchain.image(image_index);
    image_barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                          1};

    VkDependencyInfo barrier = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    barrier.imageMemoryBarrierCount = image_barriers.size();
    barrier.pImageMemoryBarriers = image_barriers.data();
    vkCmdPipelineBarrier2(cb_, &barrier);

    // Blit
    VkImageBlit2 region = {VK_STRUCTURE_TYPE_IMAGE_BLIT_2};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffsets[0] = {0, 0, 0};
    region.srcOffsets[1] = {static_cast<int32_t>(cuda_image_.width()),
                            static_cast<int32_t>(cuda_image_.height()), 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstOffsets[0] = {0, 0, 0};
    region.dstOffsets[1] = {static_cast<int32_t>(swapchain.width()),
                            static_cast<int32_t>(swapchain.height()), 1};

    VkBlitImageInfo2 blit_info = {VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2};
    blit_info.srcImage = cuda_image_.image();
    blit_info.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    blit_info.dstImage = swapchain.image(image_index);
    blit_info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    blit_info.regionCount = 1;
    blit_info.pRegions = &region;
    blit_info.filter = VK_FILTER_NEAREST;
    vkCmdBlitImage2(cb_, &blit_info);

    // Layout transition
    image_barriers[0] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    image_barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    image_barriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    image_barriers[0].dstStageMask = 0;
    image_barriers[0].dstAccessMask = 0;
    image_barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    image_barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_barriers[0].image = cuda_image_.image();
    image_barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                          1};

    image_barriers[1] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    image_barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    image_barriers[1].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    image_barriers[1].dstStageMask = 0;
    image_barriers[1].dstAccessMask = 0;
    image_barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barriers[1].newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    image_barriers[1].image = swapchain.image(image_index);
    image_barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                          1};

    barrier = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    barrier.imageMemoryBarrierCount = image_barriers.size();
    barrier.pImageMemoryBarriers = image_barriers.data();
    vkCmdPipelineBarrier2(cb_, &barrier);

    vkEndCommandBuffer(cb_);

    std::vector<VkSemaphoreSubmitInfo> wait_semaphores(2);
    wait_semaphores[0] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait_semaphores[0].semaphore = cuda_semaphore_.semaphore();
    wait_semaphores[0].stageMask = 0;

    wait_semaphores[1] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait_semaphores[1].semaphore = image_acquired_semaphore_;
    wait_semaphores[1].stageMask = 0;

    std::vector<VkCommandBufferSubmitInfo> command_buffer_submit_info(1);
    command_buffer_submit_info[0] = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    command_buffer_submit_info[0].commandBuffer = cb_;

    std::vector<VkSemaphoreSubmitInfo> signal_semaphores(1);
    signal_semaphores[0] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal_semaphores[0].semaphore = render_finished_semaphore_;
    signal_semaphores[0].stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

    VkSubmitInfo2 submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit_info.waitSemaphoreInfoCount = wait_semaphores.size();
    submit_info.pWaitSemaphoreInfos = wait_semaphores.data();
    submit_info.commandBufferInfoCount = command_buffer_submit_info.size();
    submit_info.pCommandBufferInfos = command_buffer_submit_info.data();
    submit_info.signalSemaphoreInfoCount = signal_semaphores.size();
    submit_info.pSignalSemaphoreInfos = signal_semaphores.data();
    vkQueueSubmit2(context_.queue(), 1, &submit_info, fence_);

    VkSwapchainKHR swapchain_handle = swapchain.swapchain();
    VkPresentInfoKHR present_info = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &render_finished_semaphore_;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain_handle;
    present_info.pImageIndices = &image_index;
    vkQueuePresentKHR(context_.queue(), &present_info);

    std::cout << "frame " << frame_index_ << std::endl;
    frame_index_++;
  }

 private:
  vk::Context context_;
  std::unordered_map<GLFWwindow*, vk::Swapchain> swapchains_;
  vk::CudaImage cuda_image_;
  vk::CudaSemaphore cuda_semaphore_;
  cudaStream_t stream_;
  void* test_cuda_mem_ = nullptr;

  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkCommandBuffer cb_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  VkSemaphore image_acquired_semaphore_ = VK_NULL_HANDLE;
  VkSemaphore render_finished_semaphore_ = VK_NULL_HANDLE;

  uint64_t frame_index_ = 0;
};

Engine::Engine() : impl_(std::make_shared<Impl>()) {}

Engine::~Engine() {}

void Engine::Draw(Window window) { impl_->Draw(window); }

}  // namespace pygs
