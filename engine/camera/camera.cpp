#include "camera.h"
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

glm::mat4 Camera::getViewMatrix() const {
    glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.f), position);
    glm::mat4 cameraRotation = getRotationMatrix();
    return glm::inverse(cameraTranslation * cameraRotation);
}


glm::mat4 Camera::getRotationMatrix() const {
    glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3(1.f, 0.f, 0.f));
    glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3(0.f, -1.f, 0.f));

    return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}

void Camera::processSDLEvent(SDL_Event &e) {
    if (e.type == SDL_EVENT_KEY_DOWN) {
        if (e.key.key == SDLK_W) {
            velocity.z = -1;
        }
        if (e.key.key == SDLK_S) {
            velocity.z = 1;
        }
        if (e.key.key == SDLK_A) {
            velocity.x = -1;
        }
        if (e.key.key == SDLK_D) {
            velocity.x = 1;
        }
    }

    if (e.type == SDL_EVENT_KEY_UP) {
        if (e.key.key == SDLK_W) {
            velocity.z = 0;
        }
        if (e.key.key == SDLK_S) {
            velocity.z = 0;
        }
        if (e.key.key == SDLK_A) {
            velocity.x = 0;
        }
        if (e.key.key == SDLK_D) {
            velocity.x = 0;
        }
    }

    if (e.type == SDL_EVENT_MOUSE_MOTION) {
        yaw += (float) e.motion.xrel / 200.f;
        pitch -= (float) e.motion.yrel / 200.f;
    }
}

void Camera::update() {
    glm::mat4 cameraRotation = getRotationMatrix();
    position += glm::vec3(cameraRotation * glm::vec4(velocity * 0.5f, 0.f));
}
