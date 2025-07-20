use ash::{vk, Device};
use std::default::Default;

use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;

use crate::vulkan_helpers::*;
use rust_test::svosdf::*;

#[derive(Clone, Copy)]
pub struct OctreeNodeGpu {
    pub bounds_min: [u32; 3],
    pub bounds_max: [u32; 3],
    pub brick_index: u32,
    pub child_mask: u32,
    pub children_offset: u32,
    pub is_leaf: u32,
    pub _padding: [u32; 2],
}

pub struct SvoTexture {
    pub brick_texture: VkImage,
    pub brick_upload_buffer: VkBuffer,
    pub octree_buffer: VkBuffer,
    pub sampler: vk::Sampler,
    pub brick_view: vk::ImageView,
    pub brick_texture_descriptor: vk::DescriptorImageInfo,
    pub octree_buffer_descriptor: vk::DescriptorBufferInfo,
    pub total_brick_voxels: usize,
}

impl SvoTexture {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        svo_sdf: &SvoSdf,
    ) -> SvoTexture {
        // Calculate total voxels across all bricks
        let total_brick_voxels: usize = svo_sdf.bricks
            .iter()
            .map(|brick| (brick.size * brick.size * brick.size) as usize)
            .sum();

        // Create upload buffer for all brick data
        let brick_buffer_info = vk::BufferCreateInfo {
            size: (std::mem::size_of::<u16>() * total_brick_voxels) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let brick_upload_buffer = VkBuffer::new(
            device,
            allocator,
            &brick_buffer_info,
            MemoryLocation::CpuToGpu,
        );

        // Pack all brick data into the upload buffer
        let mut offset = 0;
        for brick in &svo_sdf.bricks {
            brick_upload_buffer.copy_from_slice(&brick.data[..], offset);
            offset += brick.data.len() * std::mem::size_of::<u16>();
        }

        // Create 3D texture array for bricks
        // We'll use a large 3D texture and pack bricks into it
        let max_brick_size = svo_sdf.bricks.iter().map(|b| b.size).max().unwrap_or(8);
        let bricks_per_row = ((total_brick_voxels as f32).cbrt() / max_brick_size as f32).ceil() as u32;
        let texture_size = bricks_per_row * max_brick_size;

        let texture_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_3D,
            format: vk::Format::R16_UNORM,
            extent: vk::Extent3D {
                width: texture_size,
                height: texture_size,
                depth: texture_size,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let brick_texture = VkImage::new(
            device,
            allocator,
            &texture_create_info,
            MemoryLocation::GpuOnly,
        );

        // Create octree structure buffer
        let octree_nodes = Self::flatten_octree(&svo_sdf.root);
        let octree_buffer_info = vk::BufferCreateInfo {
            size: (std::mem::size_of::<OctreeNodeGpu>() * octree_nodes.len()) as u64,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let octree_buffer = VkBuffer::new(
            device,
            allocator,
            &octree_buffer_info,
            MemoryLocation::CpuToGpu,
        );

        octree_buffer.copy_from_slice(&octree_nodes[..], 0);

        let octree_buffer_descriptor = vk::DescriptorBufferInfo {
            buffer: octree_buffer.buffer,
            offset: 0,
            range: (std::mem::size_of::<OctreeNodeGpu>() * octree_nodes.len()) as u64,
        };

        let sampler_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            max_anisotropy: 1.0,
            border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
            compare_op: vk::CompareOp::NEVER,
            min_lod: 0.0,
            max_lod: 0.0,
            ..Default::default()
        };

        let sampler = unsafe { device.create_sampler(&sampler_info, None).unwrap() };

        let view_info = vk::ImageViewCreateInfo {
            view_type: vk::ImageViewType::TYPE_3D,
            format: texture_create_info.format,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            image: brick_texture.image,
            ..Default::default()
        };
        let brick_view = unsafe { device.create_image_view(&view_info, None) }.unwrap();

        let brick_texture_descriptor = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image_view: brick_view,
            sampler,
        };

        SvoTexture {
            brick_texture,
            brick_upload_buffer,
            octree_buffer,
            sampler,
            brick_view,
            brick_texture_descriptor,
            octree_buffer_descriptor,
            total_brick_voxels,
        }
    }

    fn flatten_octree(node: &OctreeNode) -> Vec<OctreeNodeGpu> {
        let mut nodes = Vec::new();
        let mut node_index = 0;
        Self::flatten_octree_recursive(node, &mut nodes, &mut node_index);
        nodes
    }

    fn flatten_octree_recursive(
        node: &OctreeNode,
        nodes: &mut Vec<OctreeNodeGpu>,
        node_index: &mut u32,
    ) {
        let current_index = *node_index;
        *node_index += 1;

        // Calculate child mask
        let mut child_mask = 0u32;
        for (i, child) in node.children.iter().enumerate() {
            if child.is_some() {
                child_mask |= 1 << i;
            }
        }

        let children_offset = if node.is_leaf { 0 } else { *node_index };

        // Add current node
        nodes.push(OctreeNodeGpu {
            bounds_min: [node.bounds.min.0, node.bounds.min.1, node.bounds.min.2],
            bounds_max: [node.bounds.max.0, node.bounds.max.1, node.bounds.max.2],
            brick_index: node.brick_index.unwrap_or(0xFFFFFFFF),
            child_mask,
            children_offset,
            is_leaf: if node.is_leaf { 1 } else { 0 },
            _padding: [0, 0],
        });

        // Recursively add children
        if !node.is_leaf {
            for child in &node.children {
                if let Some(child_node) = child {
                    Self::flatten_octree_recursive(child_node, nodes, node_index);
                }
            }
        }
    }

    pub fn gpu_setup(
        &self,
        device: &Device,
        command_buffer: &vk::CommandBuffer,
        svo_sdf: &SvoSdf,
    ) {
        // Setup brick texture
        let texture_barrier = vk::ImageMemoryBarrier {
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image: self.brick_texture.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        // Create copy regions for each brick
        let max_brick_size = svo_sdf.bricks.iter().map(|b| b.size).max().unwrap_or(8);
        let bricks_per_row = ((self.total_brick_voxels as f32).cbrt() / max_brick_size as f32).ceil() as u32;
        
        let mut image_copys = Vec::new();
        let mut buffer_offset = 0u64;

        for (i, brick) in svo_sdf.bricks.iter().enumerate() {
            let brick_x = (i as u32) % bricks_per_row;
            let brick_y = ((i as u32) / bricks_per_row) % bricks_per_row;
            let brick_z = (i as u32) / (bricks_per_row * bricks_per_row);

            let copy_region = vk::BufferImageCopy::builder()
                .buffer_offset(buffer_offset)
                .image_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .layer_count(1)
                        .build(),
                )
                .image_offset(vk::Offset3D {
                    x: (brick_x * max_brick_size) as i32,
                    y: (brick_y * max_brick_size) as i32,
                    z: (brick_z * max_brick_size) as i32,
                })
                .image_extent(vk::Extent3D {
                    width: brick.size,
                    height: brick.size,
                    depth: brick.size,
                });

            image_copys.push(copy_region.build());
            buffer_offset += (brick.data.len() * std::mem::size_of::<u16>()) as u64;
        }

        let texture_barrier_end = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image: self.brick_texture.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        unsafe {
            device.cmd_pipeline_barrier(
                *command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[texture_barrier],
            );

            device.cmd_copy_buffer_to_image(
                *command_buffer,
                self.brick_upload_buffer.buffer,
                self.brick_texture.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &image_copys[..],
            );

            device.cmd_pipeline_barrier(
                *command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[texture_barrier_end],
            );
        };
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.brick_view, None);
            self.brick_texture.destroy(device, allocator);
            self.brick_upload_buffer.destroy(device, allocator);
            self.octree_buffer.destroy(device, allocator);
            device.destroy_sampler(self.sampler, None);
        }
    }
}