#include "swapchain.h"

#include <iostream>

namespace pygs {
namespace vk {

class Swapchain::Impl {
 public:
  Impl() = delete;

  Impl(Context context, VkSurfaceKHR surface)
      : context_(context), surface_(surface) {
    usage_ =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkSurfaceCapabilitiesKHR surface_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context.physical_device(),
                                              surface_, &surface_capabilities);

    VkSwapchainCreateInfoKHR swapchain_info = {
        VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    swapchain_info.surface = surface_;
    swapchain_info.minImageCount = 3;
    swapchain_info.imageFormat = VK_FORMAT_B8G8R8A8_SRGB;
    swapchain_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swapchain_info.imageExtent = surface_capabilities.currentExtent;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = usage_;
    swapchain_info.preTransform = surface_capabilities.currentTransform;
    swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swapchain_info.clipped = VK_TRUE;
    vkCreateSwapchainKHR(context.device(), &swapchain_info, NULL, &swapchain_);
    width_ = swapchain_info.imageExtent.width;
    height_ = swapchain_info.imageExtent.height;

    uint32_t image_count = 0;
    vkGetSwapchainImagesKHR(context.device(), swapchain_, &image_count, NULL);
    images_.resize(image_count);
    vkGetSwapchainImagesKHR(context.device(), swapchain_, &image_count,
                            images_.data());

    image_views_.resize(image_count);
    for (int i = 0; i < image_count; i++) {
      VkImageViewCreateInfo image_view_info = {
          VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      image_view_info.image = images_[i];
      image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      image_view_info.format = swapchain_info.imageFormat;
      image_view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                          1};
      vkCreateImageView(context.device(), &image_view_info, NULL,
                        &image_views_[i]);
    }
  }

  ~Impl() {
    for (auto image_view : image_views_) {
      vkDestroyImageView(context_.device(), image_view, NULL);
    }

    vkDestroySwapchainKHR(context_.device(), swapchain_, NULL);
    vkDestroySurfaceKHR(context_.instance(), surface_, NULL);
  }

  VkSwapchainKHR swapchain() const noexcept { return swapchain_; }
  uint32_t width() const noexcept { return width_; }
  uint32_t height() const noexcept { return height_; }
  VkImage image(int index) const { return images_[index]; }
  VkImageView image_view(int index) const { return image_views_[index]; }

  bool AcquireNextImage(VkSemaphore semaphore, uint32_t* image_index) {
    VkResult result =
        vkAcquireNextImageKHR(context_.device(), swapchain_, UINT64_MAX,
                              semaphore, NULL, image_index);

    switch (result) {
      case VK_SUCCESS:
        return true;

      case VK_SUBOPTIMAL_KHR:
        should_recreate_ = true;
        return true;

      case VK_ERROR_OUT_OF_DATE_KHR:
        should_recreate_ = true;
        return false;

      default:
        return false;
    }
  }

  bool ShouldRecreate() { return should_recreate_; }

  void Recreate() {
    VkSurfaceCapabilitiesKHR surface_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context_.physical_device(),
                                              surface_, &surface_capabilities);

    VkSwapchainCreateInfoKHR swapchain_info = {
        VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    swapchain_info.surface = surface_;
    swapchain_info.minImageCount = 3;
    swapchain_info.imageFormat = VK_FORMAT_B8G8R8A8_SRGB;
    swapchain_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swapchain_info.imageExtent = surface_capabilities.currentExtent;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = usage_;
    swapchain_info.preTransform = surface_capabilities.currentTransform;
    swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swapchain_info.clipped = VK_TRUE;
    swapchain_info.oldSwapchain = swapchain_;

    VkSwapchainKHR new_swapchain;
    vkCreateSwapchainKHR(context_.device(), &swapchain_info, NULL,
                         &new_swapchain);
    vkDestroySwapchainKHR(context_.device(), swapchain_, NULL);
    swapchain_ = new_swapchain;

    width_ = swapchain_info.imageExtent.width;
    height_ = swapchain_info.imageExtent.height;

    uint32_t image_count = 0;
    vkGetSwapchainImagesKHR(context_.device(), swapchain_, &image_count, NULL);
    images_.resize(image_count);
    vkGetSwapchainImagesKHR(context_.device(), swapchain_, &image_count,
                            images_.data());

    for (auto image_view : image_views_)
      vkDestroyImageView(context_.device(), image_view, NULL);

    image_views_.resize(image_count);
    for (int i = 0; i < image_count; i++) {
      VkImageViewCreateInfo image_view_info = {
          VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      image_view_info.image = images_[i];
      image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      image_view_info.format = swapchain_info.imageFormat;
      image_view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                          1};
      vkCreateImageView(context_.device(), &image_view_info, NULL,
                        &image_views_[i]);
    }

    should_recreate_ = false;
  }

 private:
  Context context_;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  VkImageUsageFlags usage_ = 0;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  std::vector<VkImage> images_;
  std::vector<VkImageView> image_views_;
  bool should_recreate_ = false;
};

Swapchain::Swapchain() = default;

Swapchain::Swapchain(Context context, VkSurfaceKHR surface)
    : impl_(std::make_shared<Impl>(context, surface)) {}

Swapchain::~Swapchain() = default;

VkSwapchainKHR Swapchain::swapchain() const { return impl_->swapchain(); }

uint32_t Swapchain::width() const { return impl_->width(); }

uint32_t Swapchain::height() const { return impl_->height(); }

VkImage Swapchain::image(int index) const { return impl_->image(index); }

VkImageView Swapchain::image_view(int index) const {
  return impl_->image_view(index);
}

bool Swapchain::ShouldRecreate() const { return impl_->ShouldRecreate(); }

void Swapchain::Recreate() { impl_->Recreate(); }

bool Swapchain::AcquireNextImage(VkSemaphore semaphore, uint32_t* image_index) {
  return impl_->AcquireNextImage(semaphore, image_index);
}

}  // namespace vk
}  // namespace pygs
