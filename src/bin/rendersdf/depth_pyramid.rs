use std::default::Default;
use std::ffi::CString;
use std::io::Cursor;
use std::mem;

use ash::util::*;
use ash::{vk, Device};

use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;

use crate::vulkan_helpers::*;

#[derive(Clone, Copy)]
pub struct DepthPyramidPushConstants {
    pub mip: u32,
}

#[derive(Clone, Copy)]
pub struct DepthPyramidUniforms {
    pub depth_buffer_dimensions: (u32, u32),
    pub depth_pyramid_dimension: u32, // pow2 y dimension of mip 0 (texture x is 1.5x wider)
}

pub struct DepthPyramid {
    pub pipeline_layout: vk::PipelineLayout,
    pub uniform_buffer: VkBuffer,
    pub uniform_buffer_gpu: VkBuffer,
    pub image: VkImage,
    pub image_debug: VkImage,
    pub image_counters: VkImage,
    pub view: vk::ImageView,
    pub view_debug: vk::ImageView,
    pub view_counters: vk::ImageView,
    pub sampler: vk::Sampler,
    pub descriptor_rw: vk::DescriptorImageInfo,
    pub descriptor_sample: vk::DescriptorImageInfo,
    pub descriptor_debug_rw: vk::DescriptorImageInfo,
    pub descriptor_debug_sample: vk::DescriptorImageInfo,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub compute_pipeline_pass_1: vk::Pipeline,
    pub compute_shader_module_pass_1: vk::ShaderModule,
    pub compute_pipeline_downsample: vk::Pipeline,
    pub compute_shader_module_downsample: vk::ShaderModule,
}

impl DepthPyramid {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        descriptor_pool: &vk::DescriptorPool,
        depth_view: &vk::ImageView,
        image_dimensions: (u32, u32),
    ) -> DepthPyramid {
        let uniform_buffer_info = vk::BufferCreateInfo {
            size: std::mem::size_of::<DepthPyramidUniforms>() as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let uniform_buffer = VkBuffer::new(
            device,
            allocator,
            &uniform_buffer_info,
            MemoryLocation::CpuToGpu,
        );

        let uniform_buffer_gpu_info = vk::BufferCreateInfo {
            size: std::mem::size_of::<DepthPyramidUniforms>() as u64,
            usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::UNIFORM_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let uniform_buffer_gpu = VkBuffer::new(
            device,
            allocator,
            &uniform_buffer_gpu_info,
            MemoryLocation::GpuOnly,
        );

        let image_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R32_SFLOAT,
            extent: vk::Extent3D {
                width: image_dimensions.0,
                height: image_dimensions.1,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let image = VkImage::new(
            device,
            allocator,
            &image_create_info,
            MemoryLocation::GpuOnly,
        );

        let image_debug_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R32_UINT,
            extent: vk::Extent3D {
                width: image_dimensions.0,
                height: image_dimensions.1,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let image_debug = VkImage::new(
            device,
            allocator,
            &image_debug_create_info,
            MemoryLocation::GpuOnly,
        );

        let group_dim = (8, 8);

        let image_counters_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R32_UINT,
            extent: vk::Extent3D {
                width: image_dimensions.0 / group_dim.0,
                height: image_dimensions.1 / group_dim.1,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let image_counters = VkImage::new(
            device,
            allocator,
            &image_counters_create_info,
            MemoryLocation::GpuOnly,
        );

        let sampler_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            max_anisotropy: 1.0,
            border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
            compare_op: vk::CompareOp::NEVER,
            ..Default::default()
        };

        let sampler = unsafe { device.create_sampler(&sampler_info, None).unwrap() };

        let view_info = vk::ImageViewCreateInfo {
            view_type: vk::ImageViewType::TYPE_2D,
            format: image_create_info.format,
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
            image: image.image,
            ..Default::default()
        };
        let view = unsafe { device.create_image_view(&view_info, None) }.unwrap();

        let view_debug_info = vk::ImageViewCreateInfo {
            view_type: vk::ImageViewType::TYPE_2D,
            format: image_debug_create_info.format,
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
            image: image_debug.image,
            ..Default::default()
        };
        let view_debug = unsafe { device.create_image_view(&view_debug_info, None) }.unwrap();

        let view_counters_info = vk::ImageViewCreateInfo {
            view_type: vk::ImageViewType::TYPE_2D,
            format: image_counters_create_info.format,
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
            image: image_counters.image,
            ..Default::default()
        };
        let view_counters = unsafe { device.create_image_view(&view_counters_info, None) }.unwrap();

        let descriptor_rw = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: view,
            ..Default::default()
        };

        let descriptor_sample = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: view,
            sampler,
        };

        let descriptor_debug_rw = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: view_debug,
            ..Default::default()
        };

        let descriptor_debug_sample = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: view_debug,
            sampler,
        };

        let descriptor_counters = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: view_counters,
            ..Default::default()
        };

        let desc_layout_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let descriptor_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: desc_layout_bindings.len() as u32,
            p_bindings: desc_layout_bindings.as_ptr(),
            ..Default::default()
        };

        let desc_set_layout =
            unsafe { device.create_descriptor_set_layout(&descriptor_info, None) }.unwrap();

        let desc_set_layouts = &[desc_set_layout];

        let desc_alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: *descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: desc_set_layouts.as_ptr(),
            ..Default::default()
        };

        let descriptor_sets = {
            unsafe { device.allocate_descriptor_sets(&desc_alloc_info) }.unwrap()
        };

        let uniform_buffer_descriptor = vk::DescriptorBufferInfo {
            buffer: uniform_buffer_gpu.buffer,
            offset: 0,
            range: mem::size_of::<DepthPyramidUniforms>() as u64,
        };

        let depth_image_descriptor = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image_view: *depth_view,
            sampler,
        };

        let write_desc_sets = [
            vk::WriteDescriptorSet {
                dst_set: descriptor_sets[0],
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_buffer_info: &uniform_buffer_descriptor,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: descriptor_sets[0],
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: &depth_image_descriptor,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: descriptor_sets[0],
                dst_binding: 2,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                p_image_info: &descriptor_rw,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: descriptor_sets[0],
                dst_binding: 3,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                p_image_info: &descriptor_counters,
                ..Default::default()
            },
        ];
        unsafe { device.update_descriptor_sets(&write_desc_sets, &[]) };

        let layout_create_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: desc_set_layouts.len() as u32,
            p_set_layouts: desc_set_layouts.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&layout_create_info, None) }.unwrap();

        let compute_shader_module_pass_1 = {
            let mut comp_spv_file = Cursor::new(&include_bytes!(
                "../../../shader/depth_pyramid_first_mip.spv"
            ));
            let comp_code =
                read_spv(&mut comp_spv_file).expect("Failed to read compute shader spv file");
            let comp_shader_info = vk::ShaderModuleCreateInfo {
                code_size: comp_code.len() * 4,
                p_code: comp_code.as_ptr(),
                ..Default::default()
            };

            unsafe { device.create_shader_module(&comp_shader_info, None) }
                .expect("Fragment shader module error")
        };

        let shader_entry_name = CString::new("main").unwrap();

        let compute_pipeline_info_pass_1 = vk::ComputePipelineCreateInfo {
            stage: vk::PipelineShaderStageCreateInfo {
                module: compute_shader_module_pass_1,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            layout: pipeline_layout,
            ..Default::default()
        };

        let compute_shader_module_downsample = {
            let mut comp_spv_file = Cursor::new(&include_bytes!(
                "../../../shader/depth_pyramid_downsample.spv"
            ));
            let comp_code =
                read_spv(&mut comp_spv_file).expect("Failed to read compute shader spv file");
            let comp_shader_info = vk::ShaderModuleCreateInfo {
                code_size: comp_code.len() * 4,
                p_code: comp_code.as_ptr(),
                ..Default::default()
            };

            unsafe { device.create_shader_module(&comp_shader_info, None) }
                .expect("Fragment shader module error")
        };

        let compute_pipeline_info_downsample = vk::ComputePipelineCreateInfo {
            stage: vk::PipelineShaderStageCreateInfo {
                module: compute_shader_module_downsample,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            layout: pipeline_layout,
            ..Default::default()
        };

        let compute_pipelines = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[
                    compute_pipeline_info_pass_1,
                    compute_pipeline_info_downsample,
                ],
                None,
            )
        }
        .unwrap();

        let compute_pipeline_pass_1 = compute_pipelines[0];
        let compute_pipeline_downsample = compute_pipelines[1];

        DepthPyramid {
            pipeline_layout,
            uniform_buffer,
            uniform_buffer_gpu,
            desc_set_layout,
            descriptor_sets,
            image,
            image_debug,
            image_counters,
            view,
            view_debug,
            view_counters,
            sampler,
            descriptor_rw,
            descriptor_sample,
            descriptor_debug_rw,
            descriptor_debug_sample,
            compute_pipeline_pass_1,
            compute_pipeline_downsample,
            compute_shader_module_pass_1,
            compute_shader_module_downsample,
        }
    }

    pub fn gpu_setup(&self, device: &Device, command_buffer: &vk::CommandBuffer) {
        // Transition textures to read & write layout
        let texture_barrier = vk::ImageMemoryBarrier {
            dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            new_layout: vk::ImageLayout::GENERAL,
            image: self.image.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let texture_debug_barrier = vk::ImageMemoryBarrier {
            dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            new_layout: vk::ImageLayout::GENERAL,
            image: self.image_debug.image,
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
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[texture_barrier, texture_debug_barrier],
            );
        };
    }

    pub fn update(&self, uniforms: &DepthPyramidUniforms) {
        self.uniform_buffer.copy_from_slice(&[*uniforms], 0);
    }

    pub fn gpu_draw(
        &self,
        device: &Device,
        command_buffer: &vk::CommandBuffer,
        depth_image: &vk::Image,
        pyramid_mip0_dimension: u32,
        num_mips: u32,
    ) {
        let buffer_copy_regions = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: self.uniform_buffer.size,
        };

        let buffer_barrier = vk::BufferMemoryBarrier {
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            buffer: self.uniform_buffer_gpu.buffer,
            offset: 0,
            size: buffer_copy_regions.size,
            ..Default::default()
        };

        let buffer_barrier_end = vk::BufferMemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::INDEX_READ,
            buffer: self.uniform_buffer_gpu.buffer,
            offset: 0,
            size: buffer_copy_regions.size,
            ..Default::default()
        };

        let barrier_depth_to_read = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image: *depth_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let barrier_depth_to_attachment = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_READ,
            dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            new_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            image: *depth_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let barrier_pyramid_pass = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            old_layout: vk::ImageLayout::GENERAL,
            new_layout: vk::ImageLayout::GENERAL,
            image: self.image.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        unsafe {
            // Update uniform buffer
            device.cmd_pipeline_barrier(
                *command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[buffer_barrier],
                &[],
            );

            device.cmd_copy_buffer(
                *command_buffer,
                self.uniform_buffer.buffer,
                self.uniform_buffer_gpu.buffer,
                &[buffer_copy_regions],
            );

            device.cmd_pipeline_barrier(
                *command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[buffer_barrier_end],
                &[],
            );

            // Transition depth buffer to read
            device.cmd_pipeline_barrier(
                *command_buffer,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_depth_to_read],
            );

            // First pass downsample compute shader
            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipeline_pass_1,
            );

            device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &self.descriptor_sets[..],
                &[],
            );

            let group_dim = (8, 8);
            let dim = (
                pyramid_mip0_dimension / group_dim.0,
                pyramid_mip0_dimension / group_dim.1,
            );
            device.cmd_dispatch(*command_buffer, dim.0, dim.1, 1);

            // Transition depth buffer to depth attachment
            device.cmd_pipeline_barrier(
                *command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_depth_to_attachment],
            );

            // Barrier between pyramid passes to avoid RaW hazards
            device.cmd_pipeline_barrier(
                *command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_pyramid_pass],
            );

            // Mip generation passes
            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipeline_downsample,
            );

            device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &self.descriptor_sets[..],
                &[],
            );

            for mip in 1..num_mips {
                let mip2 = 1 << mip;
                let push_constants = DepthPyramidPushConstants { mip: mip - 1 };

                device.cmd_push_constants(
                    *command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    raw_bytes(&[push_constants]),
                );

                let dim = (
                    pyramid_mip0_dimension / (group_dim.0 * mip2),
                    pyramid_mip0_dimension / (group_dim.1 * mip2),
                );
                device.cmd_dispatch(*command_buffer, dim.0, dim.1, 1);

                // Barrier between pyramid passes to avoid RaW hazards
                device.cmd_pipeline_barrier(
                    *command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_pyramid_pass],
                );
            }
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image_view(self.view_debug, None);
            device.destroy_image_view(self.view_counters, None);
            self.image.destroy(device, allocator);
            self.image_debug.destroy(device, allocator);
            self.image_counters.destroy(device, allocator);
            self.uniform_buffer.destroy(device, allocator);
            self.uniform_buffer_gpu.destroy(device, allocator);
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.desc_set_layout, None);
            device.destroy_pipeline(self.compute_pipeline_pass_1, None);
            device.destroy_pipeline(self.compute_pipeline_downsample, None);
            device.destroy_shader_module(self.compute_shader_module_pass_1, None);
            device.destroy_shader_module(self.compute_shader_module_downsample, None);
        }
    }
}
