#include <pygs/scene/camera.h>

#include <algorithm>

#include <glm/gtc/matrix_transform.hpp>

namespace pygs {

Camera::Camera() {}

Camera::~Camera() {}

void Camera::SetWindowSize(uint32_t width, uint32_t height) {
  width_ = width;
  height_ = height;
}

glm::mat4 Camera::ProjectionMatrix() const {
  float aspect = static_cast<float>(width_) / height_;
  glm::mat4 projection = glm::perspective(fovy_, aspect, near_, far_);

  // gl to vulkan projection matrix
  glm::mat4 conversion = glm::mat4(1.f);
  conversion[1][1] = -1.f;
  conversion[2][2] = 0.5f;
  conversion[3][2] = 0.5f;
  return conversion * projection;
}

glm::mat4 Camera::ViewMatrix() const {
  return glm::lookAt(Eye(), center_, glm::vec3(0.f, 1.f, 0.f));
}

glm::vec3 Camera::Eye() const {
  const auto sin_phi = std::sin(phi_);
  const auto cos_phi = std::cos(phi_);
  const auto sin_theta = std::sin(theta_);
  const auto cos_theta = std::cos(theta_);
  return center_ +
         r_ * glm::vec3(sin_phi * sin_theta, cos_phi, sin_phi * cos_theta);
}

void Camera::Rotate(float x, float y) {
  theta_ -= rotation_sensitivity_ * x;
  float eps = glm::radians(0.1f);
  phi_ =
      std::clamp(phi_ - rotation_sensitivity_ * y, eps, glm::pi<float>() - eps);
}

void Camera::Translate(float x, float y) {
  // camera = center + r (sin phi sin theta, cos phi, sin phi cos theta)
  const auto sin_phi = std::sin(phi_);
  const auto cos_phi = std::cos(phi_);
  const auto sin_theta = std::sin(theta_);
  const auto cos_theta = std::cos(theta_);
  center_ +=
      translation_sensitivity_ * r_ *
      (-x * glm::vec3(cos_theta, 0.f, -sin_theta) +
       y * glm::vec3(-cos_phi * sin_theta, sin_phi, -cos_phi * cos_theta));
}

void Camera::Zoom(float x) { r_ /= std::exp(zoom_sensitivity_ * x); }

}  // namespace pygs
