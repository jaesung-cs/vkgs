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

  WindowSize window_size() const {
    WindowSize result;
    glfwGetFramebufferSize(window_, &result.width, &result.height);
    return result;
  }

  void CreateWindow(const WindowCreateInfo& create_info) {
    // create window
    width_ = 1600;
    height_ = 900;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window_ = glfwCreateWindow(width_, height_, "vkgs", NULL, NULL);

    // Vulkan surface
    glfwCreateWindowSurface(create_info.instance, window_, NULL, &surface_);

    // file drop callback
    glfwSetWindowUserPointer(window_, this);
    glfwSetDropCallback(window_, DropCallback);

    ImGui_ImplGlfw_InitForVulkan(window_, true);
    auto init_info = GetImguiInitInfo(create_info);
    ImGui_ImplVulkan_Init(&init_info);
  }

  void DestroyWindow() {
    glfwDestroyWindow(window_);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
  }

  void RecreateUi(const WindowCreateInfo& create_info) {
    ImGui_ImplVulkan_Shutdown();
    auto init_info = GetImguiInitInfo(create_info);
    ImGui_ImplVulkan_Init(&init_info);
  }

  void Show() { glfwShowWindow(window_); }

  void PollEvents() { glfwPollEvents(); }

  bool ShouldClose() const { return glfwWindowShouldClose(window_); }

  void NewUiFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
  }

  void DrawUi(VkCommandBuffer command_buffer) {
    ImDrawData* draw_data = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer);
  }

  std::vector<std::string> ConsumeDroppedFilepaths() { return std::move(dropped_filepaths_); }

 private:
  ImGui_ImplVulkan_InitInfo GetImguiInitInfo(const WindowCreateInfo& window_info) {
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = window_info.instance;
    init_info.PhysicalDevice = window_info.physical_device;
    init_info.Device = window_info.device;
    init_info.QueueFamily = window_info.queue_family;
    init_info.Queue = window_info.queue;
    init_info.PipelineCache = window_info.pipeline_cache;
    init_info.DescriptorPool = window_info.descriptor_pool;
    init_info.Subpass = 0;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.RenderPass = window_info.render_pass;
    init_info.MSAASamples = window_info.samples;
    init_info.Allocator = VK_NULL_HANDLE;
    init_info.CheckVkResultFn = check_vk_result;
    return init_info;
  }

  void SetDroppedFilepaths(const std::vector<std::string>& filepaths) { dropped_filepaths_ = filepaths; }

  std::vector<std::string> dropped_filepaths_;

  GLFWwindow* window_ = nullptr;
  int width_ = 0;
  int height_ = 0;

  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
};

Viewer::Viewer() : impl_(std::make_shared<Impl>()) {}

Viewer::~Viewer() = default;

void Viewer::CreateWindow(const WindowCreateInfo& create_info) { impl_->CreateWindow(create_info); }

void Viewer::DestroyWindow() { impl_->DestroyWindow(); }

void Viewer::RecreateUi(const WindowCreateInfo& create_info) { impl_->RecreateUi(create_info); }

void Viewer::Show() { impl_->Show(); }

void Viewer::PollEvents() { impl_->PollEvents(); }

std::vector<std::string> Viewer::ConsumeDroppedFilepaths() { return impl_->ConsumeDroppedFilepaths(); }

bool Viewer::ShouldClose() const { return impl_->ShouldClose(); }

void Viewer::NewUiFrame() { impl_->NewUiFrame(); }

void Viewer::DrawUi(VkCommandBuffer command_buffer) { impl_->DrawUi(command_buffer); }

VkSurfaceKHR Viewer::surface() const { return impl_->surface(); }

WindowSize Viewer::window_size() const { return impl_->window_size(); }

}  // namespace viewer
}  // namespace vkgs
