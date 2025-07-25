#![allow(dead_code)]

const NUM_DESCRIPTORS_PER_TYPE: u32 = 1024;
const NUM_DESCRIPTOR_SETS: u32 = 1024;

extern crate winit;

mod instances;
mod render_grids;

use rust_test::minivector;
use rust_test::vulkan_base;
use rust_test::vulkan_helpers;

use std::time::Instant;

use ash::vk;

use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit::keyboard::PhysicalKey;

use minivector::*;

use vulkan_base::*;
use vulkan_helpers::*;

use instances::*;
use render_grids::*;

#[derive(Clone, Copy)]
pub struct Vertex {
    pub pos: [f32; 4],
    pub uv: [f32; 2],
}

fn main() {
    let diagonal = Vec3 {
        x: 150.0,
        y: 150.0,
        z: 150.0,
    };

    let center_to_edge = diagonal * 0.5;
    let diagonal_length = diagonal.length();

    // Window
    let window_width = 1920;
    let window_height = 1080;

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Vulkan Test")
        .with_inner_size(winit::dpi::PhysicalSize::new(
            f64::from(window_width),
            f64::from(window_height),
        ))
        .build(&event_loop)
        .unwrap();

    // Vulkan base initialization
    let mut base = VulkanBase::new(&window, window_width, window_height);

    // Render passes
    let render_pass_attachments = [
        vk::AttachmentDescription {
            format: base.surface_format.format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        },
        vk::AttachmentDescription {
            format: vk::Format::D32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        },
    ];
    let color_attachment_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_attachment_ref = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let dependencies = [vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ..Default::default()
    }];

    let subpasses = [vk::SubpassDescription {
        pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count: color_attachment_refs.len() as u32,
        p_color_attachments: color_attachment_refs.as_ptr(),
        p_depth_stencil_attachment: &depth_attachment_ref,
        ..Default::default()
    }];

    let render_pass_create_info = vk::RenderPassCreateInfo {
        attachment_count: render_pass_attachments.len() as u32,
        p_attachments: render_pass_attachments.as_ptr(),
        subpass_count: subpasses.len() as u32,
        p_subpasses: subpasses.as_ptr(),
        dependency_count: dependencies.len() as u32,
        p_dependencies: dependencies.as_ptr(),
        ..Default::default()
    };

    let render_pass = unsafe {
        base.device
            .create_render_pass(&render_pass_create_info, None)
    }
    .unwrap();

    let framebuffers: Vec<vk::Framebuffer> = base
        .present_image_views
        .iter()
        .map(|&present_image_view| {
            let framebuffer_attachments = [present_image_view, base.depth_image_view];
            let frame_buffer_create_info = vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: framebuffer_attachments.len() as u32,
                p_attachments: framebuffer_attachments.as_ptr(),
                width: base.surface_resolution.width,
                height: base.surface_resolution.height,
                layers: 1,
                ..Default::default()
            };
            unsafe {
                base.device
                    .create_framebuffer(&frame_buffer_create_info, None)
            }
            .unwrap()
        })
        .collect();

    let view_scissor = {
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: base.surface_resolution.width as f32,
            height: base.surface_resolution.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            extent: base.surface_resolution,
            ..Default::default()
        };
        VkViewScissor { viewport, scissor }
    };

    // Descriptor pool
    let descriptor_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: NUM_DESCRIPTORS_PER_TYPE,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: NUM_DESCRIPTORS_PER_TYPE,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: NUM_DESCRIPTORS_PER_TYPE,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: NUM_DESCRIPTORS_PER_TYPE,
        },
    ];
    let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
        pool_size_count: descriptor_sizes.len() as u32,
        p_pool_sizes: descriptor_sizes.as_ptr(),
        max_sets: NUM_DESCRIPTOR_SETS,
        ..Default::default()
    };

    let descriptor_pool = unsafe {
        base.device
            .create_descriptor_pool(&descriptor_pool_info, None)
    }
    .unwrap();

    // Grid instances
    let mut instances = Instances::new(&base.device, &mut base.allocator, diagonal_length);

    // Grid renderer
    let mut render_grids = RenderGrids::new(
        &base.device,
        &base.instance,
        &mut base.allocator,
        &descriptor_pool,
        &render_pass,
        &view_scissor,
        &instances.instances_buffer_descriptor,
        NUM_INSTANCES,
    );

    // Submit initialization command buffer before rendering starts
    base.record_submit_commandbuffer(
        0,
        base.present_queue,
        &[],
        &[],
        &[],
        |device, command_buffer| {
            // GPU setup commands
            render_grids.gpu_setup(device, &command_buffer);
        },
    );

    // Camera
    struct Camera {
        position: Vec3,
        direction: Vec3,
    }

    let mut camera = Camera {
        position: Vec3 {
            x: 0.0,
            y: 2000.0,
            z: 4000.0,
        },
        direction: Vec3 {
            x: 0.0,
            y: -0.5,
            z: -1.0,
        },
    };

    // Inputs
    #[derive(Clone, Copy)]
    struct Inputs {
        is_left_clicked: bool,
        cursor_position: (i32, i32),
        wheel_delta: f32,
        keyboard_forward: i32,
        keyboard_side: i32,
    }

    impl Default for Inputs {
        fn default() -> Inputs {
            Inputs {
                is_left_clicked: false,
                cursor_position: (0, 0),
                wheel_delta: 0.0,
                keyboard_forward: 0,
                keyboard_side: 0,
            }
        }
    }

    // Window event loop
    println!("Start window event loop");

    let mut inputs_prev: Inputs = Default::default();
    let mut inputs: Inputs = Default::default();

    let mut time_start = Instant::now();
    let mut frame = 0u32;
    let mut active_command_buffer = 0;

    let _ = event_loop.run(|event, event_loop_window_target| {
        event_loop_window_target.set_control_flow(ControlFlow::Poll);

        match event {
            Event::NewEvents(_) => {
                inputs.wheel_delta = 0.0;
            }

            Event::AboutToWait => {
                let cursor_delta = (
                    inputs.cursor_position.0 - inputs_prev.cursor_position.0,
                    inputs.cursor_position.1 - inputs_prev.cursor_position.1,
                );

                inputs_prev = inputs;

                // Update camera based in inputs
                let view_rot = view(
                    Vec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    camera.direction,
                    Vec3 {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                );

                let forward_speed = inputs.wheel_delta * 5.0 + inputs.keyboard_forward as f32 * 1.5;
                camera.position = camera.position + camera.direction * forward_speed;

                let side_speed = inputs.keyboard_side as f32 * 1.5;
                let side_vec = Vec3 {
                    x: view_rot.r0.x,
                    y: view_rot.r1.x,
                    z: view_rot.r2.x,
                };
                camera.position = camera.position + side_vec * side_speed;

                if inputs.is_left_clicked {
                    let rot = rot_y_axis(cursor_delta.0 as f32 * 0.0015)
                        * rot_x_axis(cursor_delta.1 as f32 * 0.0015);

                    let rot = rot * inverse(view_rot);

                    camera.direction = Vec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 1.0,
                    } * rot;

                    camera.direction = camera.direction.normalize();
                }

                // Render
                let (present_index, _) = unsafe {
                    base.swapchain_loader.acquire_next_image(
                        base.swapchain,
                        std::u64::MAX,
                        base.present_complete_semaphore,
                        vk::Fence::null(),
                    )
                }
                .unwrap();

                // Update uniform buffer
                let color = Vec4 {
                    x: 1.0,
                    y: 0.1,
                    z: 0.0,
                    w: 0.0,
                };

                let world_to_screen = view(
                    camera.position,
                    camera.direction,
                    Vec3 {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                ) * projection(
                    std::f32::consts::PI / 2.0,
                    window_width as f32 / window_height as f32,
                    1.0,
                    10000000.0,
                );

                let grid_uniforms = GridUniforms {
                    world_to_screen,
                    color,
                    center_to_edge: center_to_edge.to_4d(),
                };

                render_grids.update(&grid_uniforms);

                // Setup render passs
                let clear_values = [
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0,
                            stencil: 0,
                        },
                    },
                ];

                let render_pass_begin_info = vk::RenderPassBeginInfo {
                    render_pass,
                    framebuffer: framebuffers[present_index as usize],
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: base.surface_resolution,
                    },
                    clear_value_count: clear_values.len() as u32,
                    p_clear_values: clear_values.as_ptr(),
                    ..Default::default()
                };

                // Submit main command buffer
                active_command_buffer = base.record_submit_commandbuffer(
                    active_command_buffer,
                    base.present_queue,
                    &[vk::PipelineStageFlags::BOTTOM_OF_PIPE],
                    &[base.present_complete_semaphore],
                    &[base.rendering_complete_semaphore],
                    |device, command_buffer| {
                        // Draw/setup (before main render pass)
                        render_grids.gpu_draw(device, &command_buffer);

                        // Render pass
                        unsafe {
                            device.cmd_begin_render_pass(
                                command_buffer,
                                &render_pass_begin_info,
                                vk::SubpassContents::INLINE,
                            );
                            device.cmd_set_viewport(command_buffer, 0, &[view_scissor.viewport]);
                            device.cmd_set_scissor(command_buffer, 0, &[view_scissor.scissor]);
                        }

                        // Draw (main render pass)
                        render_grids.gpu_draw_main_render_pass(device, &command_buffer);

                        unsafe {
                            device.cmd_end_render_pass(command_buffer);
                        }
                    },
                );

                // Present frame
                let present_info = vk::PresentInfoKHR {
                    wait_semaphore_count: 1,
                    p_wait_semaphores: &base.rendering_complete_semaphore,
                    swapchain_count: 1,
                    p_swapchains: &base.swapchain,
                    p_image_indices: &present_index,
                    ..Default::default()
                };

                unsafe {
                    base.swapchain_loader
                        .queue_present(base.present_queue, &present_info)
                }
                .unwrap();

                // Output performance info every 60 frames
                frame += 1;
                if (frame % 60) == 0 {
                    let time_now = Instant::now();
                    let interval = (time_now - time_start).as_millis();
                    println!("Average frame time: {} ms", interval as f32 / 60.0f32);

                    time_start = time_now;
                }
            }

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => event_loop_window_target.exit(),

                // TODO: Handle swapchain resize
                WindowEvent::Resized { .. } => {}

                // Keyboard
                WindowEvent::KeyboardInput { event, .. } => {
                    let pressed = event.state == ElementState::Pressed;
                    match &event.physical_key {
                        PhysicalKey::Code(keycode) => match keycode {
                            winit::keyboard::KeyCode::KeyW => inputs.keyboard_forward = if pressed { 1 } else { 0 },
                            winit::keyboard::KeyCode::KeyS => inputs.keyboard_forward = if pressed { -1 } else { 0 },
                            winit::keyboard::KeyCode::KeyD => inputs.keyboard_side = if pressed { 1 } else { 0 },
                            winit::keyboard::KeyCode::KeyA => inputs.keyboard_side = if pressed { -1 } else { 0 },
                            _ => {}
                        },
                        _ => {}
                    }
                }

                // Mouse
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state,
                    ..
                } => {
                    inputs.is_left_clicked = state == ElementState::Pressed;
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let position: (i32, i32) = position.into();
                    inputs.cursor_position = position;
                }
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_, v_lines),
                    ..
                } => {
                    inputs.wheel_delta += v_lines;
                }
                _ => (),
            },

            // No LoopDestroyed in winit 0.29.x
            _ => (),
        }
    });

    println!("End window event loop");

    unsafe { base.device.device_wait_idle() }.unwrap();

    // Cleanup
    instances.destroy(&base.device, &mut base.allocator);
    render_grids.destroy(&base.device, &mut base.allocator);
    unsafe {
        base.device.destroy_descriptor_pool(descriptor_pool, None);
        for framebuffer in framebuffers {
            base.device.destroy_framebuffer(framebuffer, None);
        }
        base.device.destroy_render_pass(render_pass, None);
    }
}
