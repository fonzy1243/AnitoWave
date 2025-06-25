#include <stb/stb_image.h>
#include <iostream>
#include "vk_loader.h"

#include "../engine/anito_wave.h"
#include "../initializers/vk_initializers.h"
#include "../types/vk_types.h"
#include <glm/gtx/quaternion.hpp>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

std::optional<AllocatedImage> load_image(AnitoWave* engine, fastgltf::Asset& asset, fastgltf::Image& image) {
    AllocatedImage newImage{};

    int width, height, nrChannels;


    std::visit(fastgltf::visitor{
                       [](auto &arg) {},
                       [&](fastgltf::sources::URI &filePath) {
                           assert(filePath.fileByteOffset == 0);
                           assert(filePath.uri.isLocalPath());

                           const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());
                           unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
                           if (data) {
                               vk::Extent3D imagesize;
                               imagesize.width = width;
                               imagesize.height = height;
                               imagesize.depth = 1;

                               newImage = engine->create_image(data, imagesize, vk::Format::eR8G8B8A8Unorm,
                                                               vk::ImageUsageFlagBits::eSampled, false);

                               stbi_image_free(data);
                           }
                       },
                       [&](fastgltf::sources::Vector &vector) {
                           unsigned char *data = stbi_load_from_memory(
                                   reinterpret_cast<unsigned char *>(vector.bytes.data()),
                                   static_cast<int>(vector.bytes.size()), &width, &height, &nrChannels, 4);
                           if (data) {
                               vk::Extent3D imagesize;
                               imagesize.width = width;
                               imagesize.height = height;
                               imagesize.depth = 1;

                               newImage = engine->create_image(data, imagesize, vk::Format::eR8G8B8A8Unorm,
                                                               vk::ImageUsageFlagBits::eSampled, false);

                               stbi_image_free(data);
                           }
                       },
                       [&](fastgltf::sources::BufferView &view) {
                           auto &bufferView = asset.bufferViews[view.bufferViewIndex];
                           auto &buffer = asset.buffers[bufferView.bufferIndex];
                           std::visit(
                                   fastgltf::visitor{[](auto &arg) { fmt::println("Unexpected buffer source type!"); },
                                                     [&](fastgltf::sources::Array &blob) {
                                                         unsigned char *data = stbi_load_from_memory(
                                                                 reinterpret_cast<stbi_uc const *>(
                                                                         blob.bytes.data() + bufferView.byteOffset),
                                                                 static_cast<int>(bufferView.byteLength), &width,
                                                                 &height, &nrChannels, 4);
                                                         if (data) {
                                                             vk::Extent3D imagesize;
                                                             imagesize.width = width;
                                                             imagesize.height = height;
                                                             imagesize.depth = 1;

                                                             newImage = engine->create_image(
                                                                     data, imagesize, vk::Format::eR8G8B8A8Unorm,
                                                                     vk::ImageUsageFlagBits::eSampled, false);

                                                             stbi_image_free(data);
                                                         }
                                                     },
                                                     [&](fastgltf::sources::Vector &vector) {
                                                         unsigned char *data = stbi_load_from_memory(
                                                                 reinterpret_cast<unsigned char *>(
                                                                         vector.bytes.data() + bufferView.byteOffset),
                                                                 static_cast<int>(bufferView.byteLength), &width,
                                                                 &height, &nrChannels, 4);
                                                         if (data) {
                                                             vk::Extent3D imagesize;
                                                             imagesize.width = width;
                                                             imagesize.height = height;
                                                             imagesize.depth = 1;

                                                             newImage = engine->create_image(
                                                                     data, imagesize, vk::Format::eR8G8B8A8Unorm,
                                                                     vk::ImageUsageFlagBits::eSampled, false);

                                                             stbi_image_free(data);
                                                         }
                                                     }},
                                   buffer.data);
                       },
               },
               image.data);

    if (newImage.image == VK_NULL_HANDLE) {
        return {};
    }

    return newImage;
}

// TODO: Finish porting over vk_loader functions and methods.