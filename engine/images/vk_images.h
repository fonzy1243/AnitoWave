#ifndef VK_IMAGES_H
#define VK_IMAGES_H

#pragma once

#include <vulkan/vulkan.hpp>

namespace vkutil {
    void transition_image(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout currentLayout, vk::ImageLayout newLayout);

    void copy_image_to_image(vk::CommandBuffer cmd, vk::Image source, vk::Image destination, vk::Extent2D srcSize, vk::Extent2D dstSize);

    void generate_mipmaps(vk::CommandBuffer cmd, vk::Image image, vk::Extent2D imageSize);
}

#endif //VK_IMAGES_H
