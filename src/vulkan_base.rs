// Contains a heavily modified version of the Ash Vulkan example ExampleBase class
// Source: https://github.com/MaikKlein/ash/blob/master/examples/src/lib.rs
// License: https://github.com/MaikKlein/ash

extern crate ash;
extern crate winit;

use crate::vulkan_helpers::*;

use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;

use ash::vk;
use winit::window::Window;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use ash::{Entry};
pub use ash::{Device, Instance};
use std::borrow::Cow;
use std::default::Default;
use std::ffi::{CStr, CString};
use std::mem::ManuallyDrop;
use std::ops::Drop;
use ash::khr::surface::Instance as Surface;
use ash::khr::swapchain::Device as Swapchain;
use ash::ext::debug_utils::Instance as DebugUtils;

const NUM_COMMAND_BUFFERS: u32 = 3;

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}

pub struct CommandBuffer {
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

pub struct CommandBufferPool {
    pub pool: vk::CommandPool,
    pub command_buffers: Vec<CommandBuffer>,
}

impl CommandBufferPool {
    pub fn new(
        device: &Device,
        queue_family_index: u32,
        num_command_buffers: u32,
    ) -> CommandBufferPool {
        unsafe {
            let pool_create_info = vk::CommandPoolCreateInfo {
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                queue_family_index,
                ..Default::default()
            };

            let pool = device.create_command_pool(&pool_create_info, None).unwrap();

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
                command_buffer_count: num_command_buffers,
                command_pool: pool,
                level: vk::CommandBufferLevel::PRIMARY,
                ..Default::default()
            };

            let command_buffers = device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap();

            let fence_info = vk::FenceCreateInfo {
                flags: vk::FenceCreateFlags::SIGNALED,
                ..Default::default()
            };

            let command_buffers: Vec<CommandBuffer> = command_buffers
                .iter()
                .map(|&command_buffer| {
                    let fence = device.create_fence(&fence_info, None).unwrap();
                    CommandBuffer {
                        command_buffer,
                        fence,
                    }
                })
                .collect();

            CommandBufferPool {
                pool,
                command_buffers,
            }
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            for command_buffer in self.command_buffers.iter() {
                device.destroy_fence(command_buffer.fence, None);
            }

            device.destroy_command_pool(self.pool, None);
        }
    }
}

pub struct VulkanBase {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub surface_loader: Surface,
    pub swapchain_loader: Swapchain,
    pub debug_utils_loader: DebugUtils,
    pub debug_call_back: vk::DebugUtilsMessengerEXT,

    pub pdevice: vk::PhysicalDevice,
    pub queue_family_index: u32,
    pub present_queue: vk::Queue,

    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,

    pub swapchain: vk::SwapchainKHR,
    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,

    pub depth_image: VkImage,
    pub depth_image_view: vk::ImageView,

    pub present_complete_semaphore: vk::Semaphore,
    pub rendering_complete_semaphore: vk::Semaphore,

    pub command_buffer_pool: CommandBufferPool,

    pub allocator: ManuallyDrop<Allocator>,
}

impl VulkanBase {
    pub fn new(window: &Window, window_width: u32, window_height: u32) -> Self {
        unsafe {
            let entry = Entry::load().unwrap();
            let app_name = CString::new("VulkanTest").unwrap();

            let layer_names = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
            let layers_names_raw: Vec<*const i8> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let display_handle = window.display_handle().unwrap();
            let window_handle = window.window_handle().unwrap();
            let raw_display_handle = display_handle.as_raw();
            let raw_window_handle = window_handle.as_raw();
            let mut extension_names_raw: Vec<*const i8> = ash_window::enumerate_required_extensions(raw_display_handle)
                .unwrap()
                .to_vec();
            extension_names_raw.push(ash::ext::debug_utils::NAME.as_ptr());
            extension_names_raw.push(
                ::std::ffi::CStr::from_bytes_with_nul(b"VK_KHR_get_physical_device_properties2\0")
                    .expect("Wrong extension string")
                    .as_ptr(),
            );

            let appinfo = vk::ApplicationInfo {
                p_application_name: app_name.as_ptr(),
                application_version: 0,
                p_engine_name: app_name.as_ptr(),
                engine_version: 0,
                api_version: vk::make_api_version(0, 1, 0, 0),
                ..Default::default()
            };

            let create_info = vk::InstanceCreateInfo {
                p_application_info: &appinfo,
                pp_enabled_layer_names: layers_names_raw.as_ptr(),
                pp_enabled_extension_names: extension_names_raw.as_ptr(),
                enabled_layer_count: layers_names_raw.len() as u32,
                enabled_extension_count: extension_names_raw.len() as u32,
                ..Default::default()
            };

            let instance: Instance = entry
                .create_instance(&create_info, None)
                .expect("Instance creation error");

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT {
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                pfn_user_callback: Some(vulkan_debug_callback),
                ..Default::default()
            };

            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();
            let surface = {
                ash_window::create_surface(
                    &entry,
                    &instance,
                    raw_display_handle,
                    raw_window_handle,
                    None,
                ).unwrap()
            };
            let pdevices = instance
                .enumerate_physical_devices()
                .expect("Physical device error");
            let surface_loader = Surface::new(&entry, &instance);
            let (pdevice, queue_family_index) = pdevices
                .iter()
                .flat_map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .filter_map(|(index, info)| {
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *pdevice,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap();
                            if supports_graphic_and_surface {
                                Some((*pdevice, index))
                            } else {
                                None
                            }
                        })
                        .next()
                })
                //.skip(1)      // Enable to select secondary GPU
                .next()
                .expect("Couldn't find suitable device.");
            let queue_family_index = queue_family_index as u32;

            let device_extension_names = [
                ash::khr::swapchain::NAME, /*, &CString::new("VK_NV_mesh_shader").unwrap()*/
            ];
            let device_extension_names_raw: Vec<*const i8> = device_extension_names
                .iter()
                .map(|cstr| cstr.as_ptr())
                .collect();

            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                //geometry_shader: 1,
                ..Default::default()
            };

            //let mut mesh_shader = vk::PhysicalDeviceMeshShaderFeaturesNV::builder().mesh_shader(true).task_shader(true).build();

            let priorities = [1.0];

            let queue_info = [vk::DeviceQueueCreateInfo {
                queue_family_index,
                p_queue_priorities: priorities.as_ptr(),
                queue_count: priorities.len() as u32,
                ..Default::default()
            }];

            let device_create_info = vk::DeviceCreateInfo {
                p_queue_create_infos: queue_info.as_ptr(),
                queue_create_info_count: queue_info.len() as u32,
                pp_enabled_extension_names: device_extension_names_raw.as_ptr(),
                enabled_extension_count: device_extension_names_raw.len() as u32,
                p_enabled_features: &features,
                ..Default::default()
            };

            let device: Device = instance
                .create_device(pdevice, &device_create_info, None)
                .unwrap();

            let present_queue = device.get_device_queue(queue_family_index as u32, 0);

            let surface_formats = surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap();
            let surface_format = surface_formats
                .iter()
                .map(|sfmt| match sfmt.format {
                    vk::Format::UNDEFINED => vk::SurfaceFormatKHR {
                        format: vk::Format::B8G8R8_UNORM,
                        color_space: sfmt.color_space,
                    },
                    _ => *sfmt,
                })
                .next()
                .expect("Unable to find suitable surface format.");
            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(pdevice, surface)
                .unwrap();
            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }
            let surface_resolution = match surface_capabilities.current_extent.width {
                std::u32::MAX => vk::Extent2D {
                    width: window_width,
                    height: window_height,
                },
                _ => surface_capabilities.current_extent,
            };
            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(pdevice, surface)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::IMMEDIATE)
                //.find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);
            let swapchain_loader = Swapchain::new(&instance, &device);

            let swapchain_create_info = vk::SwapchainCreateInfoKHR {
                surface: surface,
                min_image_count: desired_image_count,
                image_color_space: surface_format.color_space,
                image_format: surface_format.format,
                image_extent: surface_resolution,
                image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                pre_transform: pre_transform,
                composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
                present_mode: present_mode,
                clipped: vk::TRUE,
                image_array_layers: 1,
                ..Default::default()
            };

            let swapchain = swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let present_image_views: Vec<vk::ImageView> = present_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo {
                        view_type: vk::ImageViewType::TYPE_2D,
                        format: surface_format.format,
                        components: vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        },
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image: image,
                        ..Default::default()
                    };
                    device.create_image_view(&create_view_info, None).unwrap()
                })
                .collect();

            let mut allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: pdevice,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .unwrap();

            let depth_image_create_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::D32_SFLOAT,
                extent: vk::Extent3D {
                    width: surface_resolution.width,
                    height: surface_resolution.height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let depth_image = VkImage::new(
                &device,
                &mut allocator,
                &depth_image_create_info,
                MemoryLocation::GpuOnly,
            );

            let depth_image_view_info = vk::ImageViewCreateInfo {
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                image: depth_image.image,
                format: depth_image_create_info.format,
                view_type: vk::ImageViewType::TYPE_2D,
                ..Default::default()
            };

            let depth_image_view = device
                .create_image_view(&depth_image_view_info, None)
                .unwrap();

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let rendering_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            let command_buffer_pool =
                CommandBufferPool::new(&device, queue_family_index, NUM_COMMAND_BUFFERS);

            let vk = VulkanBase {
                entry,
                instance,
                device,
                queue_family_index,
                pdevice,
                surface_loader,
                surface_format,
                present_queue,
                surface_resolution,
                swapchain_loader,
                swapchain,
                present_images,
                present_image_views,
                depth_image,
                depth_image_view,
                present_complete_semaphore,
                rendering_complete_semaphore,
                surface,
                debug_call_back,
                debug_utils_loader,
                command_buffer_pool,
                allocator: ManuallyDrop::new(allocator),
            };

            vk.record_submit_commandbuffer(
                0,
                present_queue,
                &[],
                &[],
                &[],
                |device, setup_command_buffer| {
                    let layout_transition_barriers = vk::ImageMemoryBarrier {
                        image: vk.depth_image.image,
                        dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        new_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            level_count: 1,
                            layer_count: 1,
                            ..Default::default()
                        },
                        ..Default::default()
                    };

                    device.cmd_pipeline_barrier(
                        setup_command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[layout_transition_barriers],
                    );
                },
            );

            vk
        }
    }

    pub fn record_submit_commandbuffer<F: FnOnce(&Device, vk::CommandBuffer)>(
        &self,
        active_command_buffer: usize,
        submit_queue: vk::Queue,
        wait_mask: &[vk::PipelineStageFlags],
        wait_semaphores: &[vk::Semaphore],
        signal_semaphores: &[vk::Semaphore],
        f: F,
    ) -> usize {
        unsafe {
            let submit_fence =
                self.command_buffer_pool.command_buffers[active_command_buffer].fence;
            let command_buffer =
                self.command_buffer_pool.command_buffers[active_command_buffer].command_buffer;

            self.device
                .wait_for_fences(&[submit_fence], true, std::u64::MAX)
                .expect("Wait for fence failed.");

            self.device
                .reset_fences(&[submit_fence])
                .expect("Reset fences failed.");

            self.device
                .reset_command_buffer(
                    command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Reset command buffer failed.");

            let command_buffer_begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };

            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Begin commandbuffer");
            f(&self.device, command_buffer);
            self.device
                .end_command_buffer(command_buffer)
                .expect("End commandbuffer");

            let command_buffers = vec![command_buffer];

            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: wait_semaphores.len() as u32,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                p_wait_dst_stage_mask: wait_mask.as_ptr(),
                command_buffer_count: command_buffers.len() as u32,
                p_command_buffers: command_buffers.as_ptr(),
                signal_semaphore_count: signal_semaphores.len() as u32,
                p_signal_semaphores: signal_semaphores.as_ptr(),
                ..Default::default()
            };

            self.device
                .queue_submit(submit_queue, &[submit_info], submit_fence)
                .expect("queue submit failed.");
        }

        let next_command_buffer = active_command_buffer + 1;
        if next_command_buffer < self.command_buffer_pool.command_buffers.len() {
            next_command_buffer
        } else {
            0
        }
    }
}

impl Drop for VulkanBase {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device
                .destroy_semaphore(self.present_complete_semaphore, None);
            self.device
                .destroy_semaphore(self.rendering_complete_semaphore, None);

            self.command_buffer_pool.destroy(&self.device);

            self.device.destroy_image_view(self.depth_image_view, None);
            self.depth_image.destroy(&self.device, &mut self.allocator);

            for &image_view in self.present_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);

            ManuallyDrop::drop(&mut self.allocator);

            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);
            self.instance.destroy_instance(None);
        }
    }
}
