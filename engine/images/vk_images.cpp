#include "vk_images.h"
#include "../initializers/vk_initializers.h"

#include <stb/stb_image.h>

void vkutil::transition_image(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout currentLayout, vk::ImageLayout newLayout) {
    vk::ImageAspectFlags aspectMask = (newLayout == vk::ImageLayout::eDepthAttachmentOptimal) ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;

    vk::ImageMemoryBarrier2 imageBarrier = vk::ImageMemoryBarrier2()
    .setPNext(nullptr)
    .setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
    .setSrcAccessMask(vk::AccessFlagBits2::eMemoryWrite)
    .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
    .setDstAccessMask(vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead)
    .setOldLayout(currentLayout)
    .setNewLayout(newLayout)
    .setSubresourceRange(vkinit::image_subresource_range(aspectMask))
    .setImage(image);

    vk::DependencyInfo depInfo = vk::DependencyInfo()
    .setPNext(nullptr)
    .setImageMemoryBarrierCount(1)
    .setPImageMemoryBarriers(&imageBarrier);

    cmd.pipelineBarrier2(&depInfo);
}

void vkutil::copy_image_to_image(vk::CommandBuffer cmd, vk::Image source, vk::Image destination, vk::Extent2D srcSize, vk::Extent2D dstSize) {
    vk::ImageBlit2 blitRegion = vk::ImageBlit2();

    blitRegion.srcOffsets[1].x = srcSize.width;
    blitRegion.srcOffsets[1].y = srcSize.height;
    blitRegion.srcOffsets[1].z = 1;

    blitRegion.dstOffsets[1].x = dstSize.width;
    blitRegion.dstOffsets[1].y = dstSize.height;
    blitRegion.dstOffsets[1].z = 1;

    blitRegion.srcSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor);
    blitRegion.srcSubresource.setBaseArrayLayer(0);
    blitRegion.srcSubresource.setLayerCount(1);
    blitRegion.srcSubresource.setMipLevel(0);

    blitRegion.dstSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor);
    blitRegion.dstSubresource.setBaseArrayLayer(0);
    blitRegion.dstSubresource.setLayerCount(1);
    blitRegion.dstSubresource.setMipLevel(0);

    vk::BlitImageInfo2 blitInfo = vk::BlitImageInfo2()
    .setDstImage(destination)
    .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
    .setSrcImage(source)
    .setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
    .setFilter(vk::Filter::eLinear)
    .setRegionCount(1)
    .setPRegions(&blitRegion);

    cmd.blitImage2(&blitInfo);
}

void vkutil::generate_mipmaps(vk::CommandBuffer cmd, vk::Image image, vk::Extent2D imageSize) {
    int mipLevels = int(std::floor(std::log2(std::max(imageSize.width, imageSize.height)))) + 1;
    for (int mip = 0; mip < mipLevels; mip++) {
        vk::Extent2D halfSize = imageSize;
        halfSize.width /= 2;
        halfSize.height /= 2;

        vk::ImageAspectFlags aspectMask = vk::ImageAspectFlagBits::eColor;

        vk::ImageMemoryBarrier2 imageBarrier = vk::ImageMemoryBarrier2()
        .setPNext(nullptr)
        .setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setSrcAccessMask(vk::AccessFlagBits2::eMemoryWrite)
        .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
        .setDstAccessMask(vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead)
        .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
        .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
        .setSubresourceRange(vkinit::image_subresource_range(aspectMask))
        .setImage(image);

        imageBarrier.subresourceRange.setLevelCount(1);
        imageBarrier.subresourceRange.setBaseMipLevel(mip);

        vk::DependencyInfo depInfo = vk::DependencyInfo()
        .setPNext(nullptr)
        .setImageMemoryBarrierCount(1)
        .setPImageMemoryBarriers(&imageBarrier);

        cmd.pipelineBarrier2(&depInfo);

        if (mip < mipLevels - 1) {
            vk::ImageBlit2 blitRegion = vk::ImageBlit2().setPNext(nullptr);

            blitRegion.srcOffsets[1].x = imageSize.width;
            blitRegion.srcOffsets[1].y = imageSize.height;
            blitRegion.srcOffsets[1].z = 1;

            blitRegion.dstOffsets[1].x = halfSize.width;
            blitRegion.dstOffsets[1].y = halfSize.height;
            blitRegion.dstOffsets[1].z = 1;

            blitRegion.srcSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor);
            blitRegion.srcSubresource.setBaseArrayLayer(0);
            blitRegion.srcSubresource.setLayerCount(1);
            blitRegion.srcSubresource.setMipLevel(mip);

            blitRegion.dstSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor);
            blitRegion.dstSubresource.setBaseArrayLayer(0);
            blitRegion.dstSubresource.setLayerCount(1);
            blitRegion.dstSubresource.setMipLevel(mip + 1);

            vk::BlitImageInfo2 blitInfo = vk::BlitImageInfo2()
            .setDstImage(image)
            .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
            .setSrcImage(image)
            .setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
            .setFilter(vk::Filter::eLinear)
            .setRegionCount(1)
            .setPRegions(&blitRegion);

            cmd.blitImage2(&blitInfo);

            imageSize = halfSize;
        }
    }

    transition_image(cmd, image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}
