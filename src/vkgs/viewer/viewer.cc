#include "vkgs/viewer/viewer.h"

#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

namespace {

void check_vk_result(VkResult err) {
  if (err == 0) return;
  std::cerr << "[imgui vulkan] Error: VkResult = " << err << std::endl;
  if (err < 0) abort();
}

}  // namespace

namespace vkgs {
namespace viewer {

class Viewer::Impl {
 public:
  static void DropCallback(GLFWwindow* window, int count, const char** paths) {
    std::vector<std::string> filepaths(paths, paths + count);
    auto* impl = reinterpret_cast<Impl*>(glfwGetWindowUserPointer(window));
    impl->SetDroppedFilepaths(filepaths);
  }

 public:
  Impl() {
    if (glfwInit() == GLFW_FALSE) throw std::runtime_error("Failed to initialize glfw.");

    // Setup Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
  }

  ~Impl() {
    ImGui::DestroyContext();

    glfwTerminate();
  }

  VkSurfaceKHR surface() const noexcept { return surface_; }

  WindowSize windowSize() const {
    WindowSize result;
    glfwGetFramebufferSize(window_, &result.width, &result.height);
    return result;
  }

  DisplayMode displayMode() const noexcept { return displayMode_; }

  void PrepareWindow(VkInstance instance) {
    instance_ = instance;

    // create window
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(1600, 900, "vkgs", NULL, NULL);

    // Vulkan surface
    glfwCreateWindowSurface(instance, window_, NULL, &surface_);

    // file drop callback
    glfwSetWindowUserPointer(window_, this);
    glfwSetDropCallback(window_, DropCallback);

    ImGui_ImplGlfw_InitForVulkan(window_, true);
  }

  void DestroyWindow() {
    if (hasUi_) {
      ImGui_ImplVulkan_Shutdown();
    }

    ImGui_ImplGlfw_Shutdown();

    vkDestroySurfaceKHR(instance_, surface_, NULL);
    glfwDestroyWindow(window_);
  }

  void PrepareUi(const UiCreateInfo& createInfo) {
    if (hasUi_) {
      ImGui_ImplVulkan_Shutdown();
    }

    auto initInfo = GetImguiInitInfo(createInfo);
    ImGui_ImplVulkan_Init(&initInfo);
    hasUi_ = true;
  }

  void SetWindowed() {
    if (displayMode_ == DisplayMode::WindowedFullscreen) {
      glfwSetWindowMonitor(window_, NULL, xpos_, ypos_, width_, height_, 0);
      displayMode_ = DisplayMode::Windowed;
    }
  }

  void SetWindowedFullscreen() {
    if (displayMode_ == DisplayMode::Windowed) {
      glfwGetWindowPos(window_, &xpos_, &ypos_);
      glfwGetWindowSize(window_, &width_, &height_);

      // TODO: multi monitor
      GLFWmonitor* primary = glfwGetPrimaryMonitor();
      const GLFWvidmode* mode = glfwGetVideoMode(primary);
      glfwSetWindowMonitor(window_, primary, 0, 0, mode->width, mode->height, mode->refreshRate);
      displayMode_ = DisplayMode::WindowedFullscreen;
    }
  }

  void SetWindowSize(int width, int height) {
    if (glfwGetWindowAttrib(window_, GLFW_MAXIMIZED)) {
      glfwRestoreWindow(window_);
    }

    glfwSetWindowSize(window_, width, height);
  }

  void PollEvents() { glfwPollEvents(); }

  bool ShouldClose() const { return glfwWindowShouldClose(window_); }

  void BeginUi() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
  }

  void EndUi() { ImGui::Render(); }

  void DrawUi(VkCommandBuffer commandBuffer) {
    ImDrawData* drawData = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(drawData, commandBuffer);
  }

  std::vector<std::string> ConsumeDroppedFilepaths() { return std::move(droppedFilepaths_); }

 private:
  ImGui_ImplVulkan_InitInfo GetImguiInitInfo(const UiCreateInfo& createInfo) {
    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.Instance = createInfo.instance;
    initInfo.PhysicalDevice = createInfo.physicalDevice;
    initInfo.Device = createInfo.device;
    initInfo.QueueFamily = createInfo.queueFamily;
    initInfo.Queue = createInfo.queue;
    initInfo.PipelineCache = createInfo.pipelineCache;
    initInfo.DescriptorPool = createInfo.descriptorPool;
    initInfo.Subpass = createInfo.subpass;
    initInfo.MinImageCount = 3;
    initInfo.ImageCount = 3;
    initInfo.RenderPass = createInfo.renderPass;
    initInfo.MSAASamples = createInfo.samples;
    initInfo.Allocator = VK_NULL_HANDLE;
    initInfo.CheckVkResultFn = check_vk_result;
    return initInfo;
  }

  void SetDroppedFilepaths(const std::vector<std::string>& filepaths) { droppedFilepaths_ = filepaths; }

  std::vector<std::string> droppedFilepaths_;

  bool hasUi_ = false;

  GLFWwindow* window_ = nullptr;
  int xpos_ = 0;
  int ypos_ = 0;
  int width_ = 0;
  int height_ = 0;
  DisplayMode displayMode_ = DisplayMode::Windowed;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
};

Viewer::Viewer() : impl_(std::make_shared<Impl>()) {}

Viewer::~Viewer() = default;

void Viewer::PrepareWindow(VkInstance instance) { impl_->PrepareWindow(instance); }

void Viewer::DestroyWindow() { impl_->DestroyWindow(); }

void Viewer::PrepareUi(const UiCreateInfo& createInfo) { impl_->PrepareUi(createInfo); }

void Viewer::SetWindowed() { impl_->SetWindowed(); }

void Viewer::SetWindowedFullscreen() { impl_->SetWindowedFullscreen(); }

void Viewer::SetWindowSize(int width, int height) { impl_->SetWindowSize(width, height); }

void Viewer::PollEvents() { impl_->PollEvents(); }

std::vector<std::string> Viewer::ConsumeDroppedFilepaths() { return impl_->ConsumeDroppedFilepaths(); }

bool Viewer::ShouldClose() const { return impl_->ShouldClose(); }

void Viewer::BeginUi() { impl_->BeginUi(); }

void Viewer::EndUi() { impl_->EndUi(); }

void Viewer::DrawUi(VkCommandBuffer commandBuffer) { impl_->DrawUi(commandBuffer); }

VkSurfaceKHR Viewer::surface() const { return impl_->surface(); }

WindowSize Viewer::windowSize() const { return impl_->windowSize(); }

DisplayMode Viewer::displayMode() const { return impl_->displayMode(); }

}  // namespace viewer
}  // namespace vkgs
