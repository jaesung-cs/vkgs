#include "pygs/engine/vulkan/cuda_image.h"

#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>

#include <vulkan/vulkan_win32.h>
#endif

#include "pygs/engine/vulkan/context.h"

namespace pygs {
namespace vk {
namespace {

cudaExternalMemory_t ImportVulkanMemoryObjectFromFileDescriptor(
    int fd, unsigned long long size, bool isDedicated) {
  cudaExternalMemory_t extMem = NULL;
  cudaExternalMemoryHandleDesc desc = {};
  memset(&desc, 0, sizeof(desc));

  desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  desc.handle.fd = fd;
  desc.size = size;
  if (isDedicated) {
    desc.flags |= cudaExternalMemoryDedicated;
  }

  cudaImportExternalMemory(&extMem, &desc);

  // Input parameter 'fd' should not be used beyond this point as CUDA has
  // assumed ownership of it
  return extMem;
}

#ifdef _WIN32
cudaExternalMemory_t ImportVulkanMemoryObjectFromNTHandle(
    HANDLE handle, unsigned long long size, bool isDedicated) {
  cudaExternalMemory_t extMem = NULL;
  cudaExternalMemoryHandleDesc desc = {};
  memset(&desc, 0, sizeof(desc));

  desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  desc.handle.win32.handle = handle;
  desc.size = size;
  if (isDedicated) {
    desc.flags |= cudaExternalMemoryDedicated;
  }

  cudaImportExternalMemory(&extMem, &desc);

  // Input parameter 'handle' should be closed if it's not needed anymore
  CloseHandle(handle);

  return extMem;
}
#endif

void* MapBufferOntoExternalMemory(cudaExternalMemory_t extMem,
                                  unsigned long long offset,
                                  unsigned long long size) {
  void* ptr = NULL;

  cudaExternalMemoryBufferDesc desc = {};
  memset(&desc, 0, sizeof(desc));
  desc.offset = offset;
  desc.size = size;
  cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);

  // Note: ‘ptr’ must eventually be freed using cudaFree()
  return ptr;
}

}  // namespace

class CudaImage::Impl {
 public:
  Impl() = delete;

  Impl(Context context, uint32_t width, uint32_t height)
      : context_(context), width_(width), height_(height) {
#ifdef _WIN32
    constexpr VkExternalMemoryHandleTypeFlagBits handle_type =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    constexpr VkExternalMemoryHandleTypeFlagBits handle_type =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    VkExternalMemoryImageCreateInfo external_image_info = {
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    external_image_info.handleTypes = handle_type;

    VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.pNext = &external_image_info;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image_info.extent = {width, height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_LINEAR;
    image_info.usage =
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(context.device(), &image_info, NULL, &image_);

    // Memory
    // TODO: allocate large memory
    VkExportMemoryAllocateInfo external_memory_info = {
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    external_memory_info.handleTypes = handle_type;

    const VkDeviceSize size =
        static_cast<VkDeviceSize>(width) * height * 4 * sizeof(float);
    VkMemoryAllocateInfo memory_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memory_info.pNext = &external_memory_info;
    memory_info.memoryTypeIndex = 0;  // TODO
    memory_info.allocationSize = size;
    vkAllocateMemory(context_.device(), &memory_info, NULL, &memory_);

    vkBindImageMemory(context.device(), image_, memory_, 0);

#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR handle_info = {
        VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    handle_info.memory = memory_;
    handle_info.handleType = handle_type;
    HANDLE handle;
    context_.GetMemoryWin32HandleKHR(&handle_info, &handle);

    cudaExternalMemory_t ext_mem =
        ImportVulkanMemoryObjectFromNTHandle(handle, size, false);
#else
    VkMemoryGetFdInfoKHR fd_info = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    fd_info.memory = memory_;
    fd_info.handleType = handle_type;
    int fd = -1;
    context_.GetMemoryFdKHR(&fd_info, &fd);

    cudaExternalMemory_t ext_mem =
        ImportVulkanMemoryObjectFromFileDescriptor(fd, size, false);
#endif

    map_ = MapBufferOntoExternalMemory(ext_mem, 0, size);
  }

  ~Impl() {
    vkDestroyImage(context_.device(), image_, NULL);
    vkFreeMemory(context_.device(), memory_, NULL);
    cudaFree(map_);
  }

  VkImage image() const noexcept { return image_; }
  uint32_t width() const noexcept { return width_; }
  uint32_t height() const noexcept { return height_; }
  void* map() noexcept { return map_; }
  const void* map() const noexcept { return map_; }

 private:
  Context context_;
  VkImage image_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;

  uint32_t width_ = 0;
  uint32_t height_ = 0;
  void* map_ = nullptr;
};

CudaImage::CudaImage() = default;

CudaImage::CudaImage(Context context, uint32_t width, uint32_t height)
    : impl_(std::make_shared<Impl>(context, width, height)) {}

CudaImage::~CudaImage() = default;

VkImage CudaImage::image() const { return impl_->image(); }

uint32_t CudaImage::width() const { return impl_->width(); }

uint32_t CudaImage::height() const { return impl_->height(); }

void* CudaImage::map() { return impl_->map(); }

const void* CudaImage::map() const { return impl_->map(); }

}  // namespace vk
}  // namespace pygs
