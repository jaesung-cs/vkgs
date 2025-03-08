#include "vkgs/engine/interop/cuda_semaphore.h"

#ifdef _WIN32
#include <windows.h>

#include <vulkan/vulkan_win32.h>
#endif

namespace vkgs {
namespace vk {
namespace {

cudaExternalSemaphore_t ImportVulkanSemaphoreObjectFromFileDescriptor(int fd) {
  cudaExternalSemaphore_t extSem = NULL;
  cudaExternalSemaphoreHandleDesc desc = {};
  memset(&desc, 0, sizeof(desc));

  desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  desc.handle.fd = fd;
  cudaImportExternalSemaphore(&extSem, &desc);

  // Input parameter 'fd' should not be used beyond this point as CUDA has
  // assumed ownership of it
  return extSem;
}

#ifdef _WIN32
cudaExternalSemaphore_t ImportVulkanSemaphoreObjectFromNTHandle(HANDLE handle) {
  cudaExternalSemaphore_t extSem = NULL;
  cudaExternalSemaphoreHandleDesc desc = {};
  memset(&desc, 0, sizeof(desc));

  desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
  desc.handle.win32.handle = handle;
  cudaImportExternalSemaphore(&extSem, &desc);

  // Input parameter 'handle' should be closed if it's not needed anymore
  CloseHandle(handle);

  return extSem;
}
#endif

void SignalExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream) {
  cudaExternalSemaphoreSignalParams params = {};
  memset(&params, 0, sizeof(params));
  cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

}  // namespace

class CudaSemaphore::Impl {
 public:
  Impl() = delete;

  Impl(Context context) : context_(context) {
#ifdef _WIN32
    constexpr VkExternalSemaphoreHandleTypeFlagBits handle_type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    constexpr VkExternalSemaphoreHandleTypeFlagBits handle_type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    VkExportSemaphoreCreateInfo external_semaphore_info = {VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    external_semaphore_info.handleTypes = handle_type;

    VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphore_info.pNext = &external_semaphore_info;
    vkCreateSemaphore(context.device(), &semaphore_info, NULL, &semaphore_);

#ifdef _WIN32
    VkSemaphoreGetWin32HandleInfoKHR handle_info = {VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    handle_info.semaphore = semaphore_;
    handle_info.handleType = handle_type;
    HANDLE handle;
    context_.GetSemaphoreWin32HandleKHR(&handle_info, &handle);

    cuda_semaphore_ = ImportVulkanSemaphoreObjectFromNTHandle(handle);
#else
    VkSemaphoreGetFdInfoKHR fd_info = {VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    fd_info.semaphore = semaphore_;
    fd_info.handleType = handle_type;
    int fd = -1;
    context_.GetSemaphoreFdKHR(&fd_info, &fd);

    cuda_semaphore_ = ImportVulkanSemaphoreObjectFromFileDescriptor(fd);
#endif
  }

  ~Impl() { vkDestroySemaphore(context_.device(), semaphore_, NULL); }

  VkSemaphore semaphore() const noexcept { return semaphore_; }

  void signal(cudaStream_t stream) { SignalExternalSemaphore(cuda_semaphore_, stream); }

 private:
  Context context_;
  VkSemaphore semaphore_ = VK_NULL_HANDLE;
  cudaExternalSemaphore_t cuda_semaphore_ = nullptr;
};

CudaSemaphore::CudaSemaphore() = default;

CudaSemaphore::CudaSemaphore(Context context) : impl_(std::make_shared<Impl>(context)) {}

CudaSemaphore::~CudaSemaphore() = default;

VkSemaphore CudaSemaphore::semaphore() const { return impl_->semaphore(); }

void CudaSemaphore::signal(cudaStream_t stream) { impl_->signal(stream); }

}  // namespace vk
}  // namespace vkgs
