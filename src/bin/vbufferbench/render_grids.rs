enum GridTechnique {
    Color,
    PrimId,
    NonIndexed,
    LeadingVertex,
    GetAttributeAtVertex,
    MeshShader,
}

const GRID_TECHNIQUE: GridTechnique = GridTechnique::LeadingVertex;

use ash::Instance;
use std::default::Default;
use std::ffi::CString;
use std::io::Cursor;
use std::mem;

use ash::util::*;
use ash::{vk, Device};

use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;

use crate::minivector::*;
use crate::vulkan_helpers::*;

#[derive(Clone, Copy)]
pub struct GridUniforms {
    pub world_to_screen: Mat4x4,
    pub color: Vec4,
    pub center_to_edge: Vec4,
}

pub struct RenderGrids {
    pub pipeline_layout: vk::PipelineLayout,
    pub index_buffer: VkBuffer,
    pub index_buffer_gpu: VkBuffer,
    pub uniform_buffer: VkBuffer,
    pub uniform_buffer_gpu: VkBuffer,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub graphic_pipeline: vk::Pipeline,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub vertex_shader_module: vk::ShaderModule,
    pub fragment_shader_module: vk::ShaderModule,
    // TODO: Re-enable mesh_shader field if/when ash exposes MeshShader extension loader
    // pub mesh_shader: ash::extensions::nv::MeshShader,
    pub num_instances: usize,
}

impl RenderGrids {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        _instance: &Instance,
        allocator: &mut Allocator,
        descriptor_pool: &vk::DescriptorPool,
        render_pass: &vk::RenderPass,
        view_scissor: &VkViewScissor,
        instances_buffer_descriptor: &vk::DescriptorBufferInfo,
        num_instances: usize,
    ) -> RenderGrids {
        // TODO: Re-enable mesh_shader instantiation if/when ash exposes MeshShader extension loader
        // let mesh_shader = ash::extensions::nv::MeshShader::new(instance, device);

        const GRID_DIM: usize = 7;
        const NUM_GRID_INDICES: usize = GRID_DIM * GRID_DIM * 2 * 3;

        let grid_stride = if let GridTechnique::LeadingVertex = GRID_TECHNIQUE {
            (GRID_DIM + 1) * 2
        } else {
            GRID_DIM + 1
        };
        let instance_stride = if let GridTechnique::LeadingVertex = GRID_TECHNIQUE {
            GRID_DIM * (GRID_DIM + 1) * 2
        } else {
            (GRID_DIM + 1) * (GRID_DIM + 1)
        };

        let mut grid_indices: [u32; NUM_GRID_INDICES] = [0; NUM_GRID_INDICES];
        for y in 0..GRID_DIM {
            for x in 0..GRID_DIM {
                let grid = x + y * GRID_DIM;
                let vertex = (x + y * grid_stride) as u32;

                // Upper left triangle
                grid_indices[grid * 6] = vertex;
                grid_indices[grid * 6 + 1] = 1 + vertex;
                grid_indices[grid * 6 + 2] = (GRID_DIM + 1) as u32 + vertex;

                // Lower right triangle
                grid_indices[grid * 6 + 3] = (GRID_DIM + 1) as u32 + vertex;
                grid_indices[grid * 6 + 4] = 1 + vertex;
                grid_indices[grid * 6 + 5] = 1 + (GRID_DIM + 1) as u32 + vertex;
            }
        }

        let num_indices = num_instances * NUM_GRID_INDICES;

        let index_buffer_data: Vec<u32> = (0..num_indices)
            .map(|i| {
                let grid = i / NUM_GRID_INDICES;
                let grid_local = i % NUM_GRID_INDICES;
                grid_indices[grid_local] + grid as u32 * instance_stride as u32
            })
            .collect();

        let index_buffer_info = vk::BufferCreateInfo {
            size: std::mem::size_of_val(&index_buffer_data[..]) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let index_buffer = VkBuffer::new(
            device,
            allocator,
            &index_buffer_info,
            MemoryLocation::CpuToGpu,
        );
        index_buffer.copy_from_slice(&index_buffer_data[..], 0);

        let index_buffer_gpu_info = vk::BufferCreateInfo {
            size: std::mem::size_of_val(&index_buffer_data[..]) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let index_buffer_gpu = VkBuffer::new(
            device,
            allocator,
            &index_buffer_gpu_info,
            MemoryLocation::GpuOnly,
        );

        let uniform_buffer_info = vk::BufferCreateInfo {
            size: std::mem::size_of::<GridUniforms>() as u64,
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
            size: std::mem::size_of::<GridUniforms>() as u64,
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

        let geom_shader_stage: vk::ShaderStageFlags =
            if let GridTechnique::MeshShader = GRID_TECHNIQUE {
                vk::ShaderStageFlags::MESH_NV
            } else {
                vk::ShaderStageFlags::VERTEX
            };

        let desc_layout_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT | geom_shader_stage,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT | geom_shader_stage,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT | geom_shader_stage,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT | geom_shader_stage,
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
            range: mem::size_of::<GridUniforms>() as u64,
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
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: instances_buffer_descriptor,
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

        let mut vertex_spv_file = Cursor::new(match GRID_TECHNIQUE {
            GridTechnique::Color => &include_bytes!("../../../shader/vbuffer_vert.spv")[..],
            GridTechnique::PrimId => &include_bytes!("../../../shader/vbuffer_vert.spv")[..],
            GridTechnique::NonIndexed => {
                &include_bytes!("../../../shader/vbuffer_nonindexed_vert.spv")[..]
            }
            GridTechnique::LeadingVertex => {
                &include_bytes!("../../../shader/vbuffer_leadingvertex_vert.spv")[..]
            }
            GridTechnique::GetAttributeAtVertex => {
                &include_bytes!("../../../shader/vbuffer_getattributeatvertex_vert.spv")[..]
            }
            GridTechnique::MeshShader => {
                &include_bytes!("../../../shader/vbuffer_meshshader_mesh.spv")[..]
            }
        });

        let mut frag_spv_file = Cursor::new(match GRID_TECHNIQUE {
            GridTechnique::Color => &include_bytes!("../../../shader/vbuffer_color_frag.spv")[..],
            GridTechnique::PrimId => &include_bytes!("../../../shader/vbuffer_primid_frag.spv")[..],
            GridTechnique::NonIndexed => {
                &include_bytes!("../../../shader/vbuffer_nonindexed_frag.spv")[..]
            }
            GridTechnique::LeadingVertex => {
                &include_bytes!("../../../shader/vbuffer_leadingvertex_frag.spv")[..]
            }
            GridTechnique::GetAttributeAtVertex => {
                &include_bytes!("../../../shader/vbuffer_getattributeatvertex_frag.spv")[..]
            }
            GridTechnique::MeshShader => {
                &include_bytes!("../../../shader/vbuffer_meshshader_frag.spv")[..]
            }
        });

        let vertex_code =
            read_spv(&mut vertex_spv_file).expect("Failed to read vertex shader spv file");
        let vertex_shader_info = vk::ShaderModuleCreateInfo {
            code_size: vertex_code.len() * 4,
            p_code: vertex_code.as_ptr(),
            ..Default::default()
        };

        let frag_code =
            read_spv(&mut frag_spv_file).expect("Failed to read fragment shader spv file");
        let frag_shader_info = vk::ShaderModuleCreateInfo {
            code_size: frag_code.len() * 4,
            p_code: frag_code.as_ptr(),
            ..Default::default()
        };

        let vertex_shader_module =
            unsafe { device.create_shader_module(&vertex_shader_info, None) }
                .expect("Vertex shader module error");

        let fragment_shader_module =
            unsafe { device.create_shader_module(&frag_shader_info, None) }
                .expect("Fragment shader module error");

        let shader_entry_name = CString::new("main").unwrap();
        let shader_stage_create_infos = [
            vk::PipelineShaderStageCreateInfo {
                module: vertex_shader_module,
                p_name: shader_entry_name.as_ptr(),
                stage: geom_shader_stage,
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                module: fragment_shader_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default();

        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let scissors = &[view_scissor.scissor];
        let viewports = &[view_scissor.viewport];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo {
            viewport_count: viewports.len() as u32,
            p_viewports: viewports.as_ptr(),
            scissor_count: scissors.len() as u32,
            p_scissors: scissors.as_ptr(),
            ..Default::default()
        };

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            ..Default::default()
        };

        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let noop_stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            ..Default::default()
        };
        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: 1,
            depth_write_enable: 1,
            depth_compare_op: vk::CompareOp::GREATER_OR_EQUAL,
            front: noop_stencil_state,
            back: noop_stencil_state,
            max_depth_bounds: 1.0,
            ..Default::default()
        };

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ZERO,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            logic_op: vk::LogicOp::CLEAR,
            attachment_count: color_blend_attachment_states.len() as u32,
            p_attachments: color_blend_attachment_states.as_ptr(),
            ..Default::default()
        };

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_state.len() as u32,
            p_dynamic_states: dynamic_state.as_ptr(),
            ..Default::default()
        };

        let graphic_pipeline_infos = vk::GraphicsPipelineCreateInfo {
            stage_count: shader_stage_create_infos.len() as u32,
            p_stages: shader_stage_create_infos.as_ptr(),
            p_vertex_input_state: &vertex_input_state_info,
            p_input_assembly_state: &vertex_input_assembly_state_info,
            p_viewport_state: &viewport_state_info,
            p_rasterization_state: &rasterization_info,
            p_multisample_state: &multisample_state_info,
            p_depth_stencil_state: &depth_state_info,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: &dynamic_state_info,
            layout: pipeline_layout,
            render_pass: *render_pass,
            ..Default::default()
        };

        let graphics_pipelines = unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[graphic_pipeline_infos],
                None,
            )
        }
        .unwrap();

        let graphic_pipeline = graphics_pipelines[0];

        RenderGrids {
            pipeline_layout,
            index_buffer,
            index_buffer_gpu,
            uniform_buffer,
            uniform_buffer_gpu,
            desc_set_layout,
            graphic_pipeline,
            descriptor_sets,
            vertex_shader_module,
            fragment_shader_module,
            // TODO: Re-enable mesh_shader field if/when ash exposes MeshShader extension loader
            // mesh_shader,
            num_instances,
        }
    }

    pub fn update(&self, uniforms: &GridUniforms) {
        self.uniform_buffer.copy_from_slice(&[*uniforms], 0);
    }

    pub fn gpu_setup(&self, device: &Device, command_buffer: &vk::CommandBuffer) {
        let buffer_copy_regions = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: self.index_buffer.size,
        };

        let buffer_barrier = vk::BufferMemoryBarrier {
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            buffer: self.index_buffer_gpu.buffer,
            offset: 0,
            size: buffer_copy_regions.size,
            ..Default::default()
        };

        let buffer_barrier_end = vk::BufferMemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::INDEX_READ,
            buffer: self.index_buffer_gpu.buffer,
            offset: 0,
            size: buffer_copy_regions.size,
            ..Default::default()
        };

        unsafe {
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
                self.index_buffer.buffer,
                self.index_buffer_gpu.buffer,
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
        };
    }

    pub fn gpu_draw(&self, device: &Device, command_buffer: &vk::CommandBuffer) {
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

        unsafe {
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
        }
    }

    pub fn gpu_draw_main_render_pass(&self, device: &Device, command_buffer: &vk::CommandBuffer) {
        unsafe {
            device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &self.descriptor_sets[..],
                &[],
            );

            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphic_pipeline,
            );

            device.cmd_bind_index_buffer(
                *command_buffer,
                self.index_buffer_gpu.buffer,
                0,
                vk::IndexType::UINT32,
            );

            match GRID_TECHNIQUE {
                GridTechnique::NonIndexed => device.cmd_draw(
                    *command_buffer,
                    self.index_buffer_gpu.size as u32 / std::mem::size_of::<u32>() as u32,
                    1,
                    0,
                    0,
                ),
                GridTechnique::MeshShader => {
                    // TODO: Re-enable mesh_shader draw call if/when ash exposes MeshShader extension loader
                },
                _ => device.cmd_draw_indexed(
                    *command_buffer,
                    self.index_buffer_gpu.size as u32 / std::mem::size_of::<u32>() as u32,
                    1,
                    0,
                    0,
                    0,
                ),
            }
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_pipeline(self.graphic_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_shader_module(self.vertex_shader_module, None);
            device.destroy_shader_module(self.fragment_shader_module, None);
            self.index_buffer.destroy(device, allocator);
            self.index_buffer_gpu.destroy(device, allocator);
            self.uniform_buffer.destroy(device, allocator);
            self.uniform_buffer_gpu.destroy(device, allocator);
            device.destroy_descriptor_set_layout(self.desc_set_layout, None);
        }
    }
}
