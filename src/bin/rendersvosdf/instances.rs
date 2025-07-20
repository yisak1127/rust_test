use std::default::Default;

use ash::{vk, Device};

use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;

use crate::minivector::*;
use crate::vulkan_helpers::*;
use rust_test::svosdf::*;

#[derive(Clone, Copy)]
pub struct InstanceData {
    pub position: Vec4,
    pub brick_index: u32,
    pub brick_size: u32,
    pub _padding: [u32; 2],
}

pub struct Instances {
    pub instances_buffer: VkBuffer,
    pub instances_buffer_descriptor: vk::DescriptorBufferInfo,
    pub num_instances: usize,
}

impl Instances {
    pub fn new(device: &Device, allocator: &mut Allocator, svo_sdf: &SvoSdf) -> Instances {
        let num_instances = svo_sdf.bricks.len();
        
        let instances_buffer_info = vk::BufferCreateInfo {
            size: (std::mem::size_of::<InstanceData>() * num_instances) as u64,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let instances_buffer = VkBuffer::new(
            device,
            allocator,
            &instances_buffer_info,
            MemoryLocation::CpuToGpu,
        );

        let instances_buffer_descriptor = vk::DescriptorBufferInfo {
            buffer: instances_buffer.buffer,
            offset: 0,
            range: (std::mem::size_of::<InstanceData>() * num_instances) as u64,
        };

        // Create instance data from SVO bricks
        let instances_buffer_data: Vec<InstanceData> = svo_sdf.bricks
            .iter()
            .enumerate()
            .map(|(i, brick)| {
                let world_pos = Vec3 {
                    x: brick.position.0 as f32 * svo_sdf.header.dx,
                    y: brick.position.1 as f32 * svo_sdf.header.dx,
                    z: brick.position.2 as f32 * svo_sdf.header.dx,
                };
                
                let brick_world_size = brick.size as f32 * svo_sdf.header.dx;
                
                InstanceData {
                    position: Vec4 {
                        x: world_pos.x,
                        y: world_pos.y,
                        z: world_pos.z,
                        w: brick_world_size,
                    },
                    brick_index: i as u32,
                    brick_size: brick.size,
                    _padding: [0, 0],
                }
            })
            .collect();

        instances_buffer.copy_from_slice(&instances_buffer_data[..], 0);

        Instances {
            instances_buffer,
            instances_buffer_descriptor,
            num_instances,
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        self.instances_buffer.destroy(device, allocator);
    }
}