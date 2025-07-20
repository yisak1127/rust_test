use crate::sdf::*;
use crate::serialization::*;
use std::io;

const LEVEL_ZERO: u16 = 65536 / 2;

#[derive(Clone, Debug)]
pub struct OctreeNode {
    pub children: [Option<Box<OctreeNode>>; 8],
    pub brick_index: Option<u32>,
    pub is_leaf: bool,
    pub bounds: BoundingBox,
}

#[derive(Clone, Debug, Copy)]
pub struct BoundingBox {
    pub min: (u32, u32, u32),
    pub max: (u32, u32, u32),
}

#[derive(Clone, Debug)]
pub struct Brick {
    pub data: Vec<u16>,
    pub size: u32,
    pub position: (u32, u32, u32),
}

pub struct SvoSdf {
    pub header: SdfHeader,
    pub root: OctreeNode,
    pub bricks: Vec<Brick>,
    pub brick_size: u32,
}

impl BoundingBox {
    pub fn new(min: (u32, u32, u32), max: (u32, u32, u32)) -> Self {
        BoundingBox { min, max }
    }

    pub fn size(&self) -> (u32, u32, u32) {
        (
            self.max.0 - self.min.0,
            self.max.1 - self.min.1,
            self.max.2 - self.min.2,
        )
    }

    pub fn center(&self) -> (u32, u32, u32) {
        (
            (self.min.0 + self.max.0) / 2,
            (self.min.1 + self.max.1) / 2,
            (self.min.2 + self.max.2) / 2,
        )
    }

    pub fn child_bounds(&self, child_index: usize) -> BoundingBox {
        let center = self.center();
        let (cx, cy, cz) = center;

        match child_index {
            0 => BoundingBox::new(self.min, center),
            1 => BoundingBox::new((cx, self.min.1, self.min.2), (self.max.0, cy, cz)),
            2 => BoundingBox::new((self.min.0, cy, self.min.2), (cx, self.max.1, cz)),
            3 => BoundingBox::new((cx, cy, self.min.2), (self.max.0, self.max.1, cz)),
            4 => BoundingBox::new((self.min.0, self.min.1, cz), (cx, cy, self.max.2)),
            5 => BoundingBox::new((cx, self.min.1, cz), (self.max.0, cy, self.max.2)),
            6 => BoundingBox::new((self.min.0, cy, cz), (cx, self.max.1, self.max.2)),
            7 => BoundingBox::new(center, self.max),
            _ => panic!("Invalid child index"),
        }
    }
}

impl OctreeNode {
    pub fn new(bounds: BoundingBox) -> Self {
        OctreeNode {
            children: [None, None, None, None, None, None, None, None],
            brick_index: None,
            is_leaf: false,
            bounds,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.brick_index.is_none() && self.children.iter().all(|child| child.is_none())
    }
}

impl Brick {
    pub fn new(size: u32, position: (u32, u32, u32)) -> Self {
        Brick {
            data: vec![LEVEL_ZERO; (size * size * size) as usize],
            size,
            position,
        }
    }

    pub fn extract_from_sdf(
        sdf: &Sdf,
        position: (u32, u32, u32),
        size: u32,
    ) -> Self {
        let mut brick = Brick::new(size, position);
        let (px, py, pz) = position;
        
        let stride_y = sdf.header.dim.0;
        let stride_z = sdf.header.dim.0 * sdf.header.dim.1;

        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let src_x = px + x;
                    let src_y = py + y;
                    let src_z = pz + z;

                    if src_x < sdf.header.dim.0 && src_y < sdf.header.dim.1 && src_z < sdf.header.dim.2 {
                        let src_index = (src_x + src_y * stride_y + src_z * stride_z) as usize;
                        let dst_index = (x + y * size + z * size * size) as usize;
                        brick.data[dst_index] = sdf.voxels[src_index];
                    }
                }
            }
        }

        brick
    }

    pub fn has_surface(&self, threshold: f32) -> bool {
        let threshold_u16 = (threshold * 65535.0) as u16;
        let mut has_inside = false;
        let mut has_outside = false;

        for &value in &self.data {
            if value < LEVEL_ZERO - threshold_u16 {
                has_inside = true;
            }
            if value > LEVEL_ZERO + threshold_u16 {
                has_outside = true;
            }
            if has_inside && has_outside {
                return true;
            }
        }

        false
    }

    pub fn is_uniform(&self, threshold: f32) -> bool {
        if self.data.is_empty() {
            return true;
        }

        let threshold_u16 = (threshold * 65535.0) as u16;
        let first_value = self.data[0];

        self.data.iter().all(|&value| {
            (value as i32 - first_value as i32).abs() <= threshold_u16 as i32
        })
    }
}

impl SvoSdf {
    pub fn from_sdf(sdf: &Sdf, brick_size: u32, max_depth: u32, threshold: f32) -> Self {
        let bounds = BoundingBox::new(
            (0, 0, 0),
            (sdf.header.dim.0, sdf.header.dim.1, sdf.header.dim.2),
        );

        let mut svo_sdf = SvoSdf {
            header: sdf.header,
            root: OctreeNode::new(bounds),
            bricks: Vec::new(),
            brick_size,
        };

        svo_sdf.build_octree(sdf, &mut svo_sdf.root, 0, max_depth, threshold);
        svo_sdf
    }

    fn build_octree(
        &mut self,
        sdf: &Sdf,
        node: &mut OctreeNode,
        depth: u32,
        max_depth: u32,
        threshold: f32,
    ) {
        let bounds_size = node.bounds.size();
        let min_size = self.brick_size;

        // If we've reached maximum depth or the node is small enough, create a leaf
        if depth >= max_depth || (bounds_size.0 <= min_size && bounds_size.1 <= min_size && bounds_size.2 <= min_size) {
            let brick = Brick::extract_from_sdf(sdf, node.bounds.min, self.brick_size.min(bounds_size.0.max(bounds_size.1.max(bounds_size.2))));
            
            // Only store the brick if it contains surface data or is not uniform
            if brick.has_surface(threshold) || !brick.is_uniform(threshold) {
                node.brick_index = Some(self.bricks.len() as u32);
                self.bricks.push(brick);
            }
            node.is_leaf = true;
            return;
        }

        // Check if this region contains any surface data
        let test_brick = Brick::extract_from_sdf(sdf, node.bounds.min, bounds_size.0.min(bounds_size.1.min(bounds_size.2)));
        if !test_brick.has_surface(threshold) && test_brick.is_uniform(threshold) {
            // This region is uniform and doesn't contain surface, so we can skip it
            return;
        }

        // Subdivide into 8 children
        for i in 0..8 {
            let child_bounds = node.bounds.child_bounds(i);
            let mut child_node = OctreeNode::new(child_bounds);
            
            self.build_octree(sdf, &mut child_node, depth + 1, max_depth, threshold);
            
            if !child_node.is_empty() {
                node.children[i] = Some(Box::new(child_node));
            }
        }
    }

    pub fn calculate_memory_usage(&self) -> usize {
        let node_size = std::mem::size_of::<OctreeNode>();
        let brick_size = self.bricks.iter().map(|b| b.data.len() * std::mem::size_of::<u16>()).sum::<usize>();
        let header_size = std::mem::size_of::<SdfHeader>();
        
        // Estimate node count (this is approximate)
        let estimated_nodes = self.count_nodes(&self.root);
        
        header_size + estimated_nodes * node_size + brick_size
    }

    fn count_nodes(&self, node: &OctreeNode) -> usize {
        let mut count = 1;
        for child in &node.children {
            if let Some(child_node) = child {
                count += self.count_nodes(child_node);
            }
        }
        count
    }

    pub fn save(&self, filename: &str) -> io::Result<()> {
        let mut storer = StorerVec::new();
        
        // Store header
        storer.store_u32(self.header.dim.0);
        storer.store_u32(self.header.dim.1);
        storer.store_u32(self.header.dim.2);
        storer.store_f32(self.header.box_min.0);
        storer.store_f32(self.header.box_min.1);
        storer.store_f32(self.header.box_min.2);
        storer.store_f32(self.header.dx);
        storer.store_u32(self.brick_size);
        
        // Store bricks
        storer.store_u32(self.bricks.len() as u32);
        for brick in &self.bricks {
            storer.store_u32(brick.size);
            storer.store_u32(brick.position.0);
            storer.store_u32(brick.position.1);
            storer.store_u32(brick.position.2);
            storer.store_array_u16(&brick.data);
        }
        
        // Store octree structure
        self.serialize_node(&self.root, &mut storer);
        
        std::fs::write(filename, storer.v)?;
        Ok(())
    }

    fn serialize_node(&self, node: &OctreeNode, storer: &mut StorerVec) {
        storer.store_u8(if node.is_leaf { 1 } else { 0 });
        
        if let Some(brick_index) = node.brick_index {
            storer.store_u8(1); // has brick
            storer.store_u32(brick_index);
        } else {
            storer.store_u8(0); // no brick
        }
        
        // Store bounds
        storer.store_u32(node.bounds.min.0);
        storer.store_u32(node.bounds.min.1);
        storer.store_u32(node.bounds.min.2);
        storer.store_u32(node.bounds.max.0);
        storer.store_u32(node.bounds.max.1);
        storer.store_u32(node.bounds.max.2);
        
        if !node.is_leaf {
            // Store child mask
            let mut child_mask = 0u8;
            for (i, child) in node.children.iter().enumerate() {
                if child.is_some() {
                    child_mask |= 1 << i;
                }
            }
            storer.store_u8(child_mask);
            
            // Recursively store children
            for child in &node.children {
                if let Some(child_node) = child {
                    self.serialize_node(child_node, storer);
                }
            }
        }
    }

    pub fn load(filename: &str) -> io::Result<Self> {
        let bytes = std::fs::read(filename)?;
        let mut loader = Loader::new();
        
        // Load header
        let header = SdfHeader {
            dim: (
                loader.load_u32(&bytes),
                loader.load_u32(&bytes),
                loader.load_u32(&bytes),
            ),
            box_min: (
                loader.load_f32(&bytes),
                loader.load_f32(&bytes),
                loader.load_f32(&bytes),
            ),
            dx: loader.load_f32(&bytes),
        };
        let brick_size = loader.load_u32(&bytes);
        
        // Load bricks
        let brick_count = loader.load_u32(&bytes);
        let mut bricks = Vec::with_capacity(brick_count as usize);
        
        for _ in 0..brick_count {
            let size = loader.load_u32(&bytes);
            let position = (
                loader.load_u32(&bytes),
                loader.load_u32(&bytes),
                loader.load_u32(&bytes),
            );
            let data = loader.load_array_u16(&bytes, (size * size * size) as usize);
            
            bricks.push(Brick {
                data,
                size,
                position,
            });
        }
        
        // Load octree structure
        let bounds = BoundingBox::new((0, 0, 0), header.dim);
        let root = Self::deserialize_node(&mut loader, &bytes, bounds);
        
        Ok(SvoSdf {
            header,
            root,
            bricks,
            brick_size,
        })
    }

    fn deserialize_node(loader: &mut Loader, bytes: &[u8], bounds: BoundingBox) -> OctreeNode {
        let is_leaf = loader.load_u8(bytes) != 0;
        
        let brick_index = if loader.load_u8(bytes) != 0 {
            Some(loader.load_u32(bytes))
        } else {
            None
        };
        
        // Load bounds (though we could reconstruct them)
        let _min_x = loader.load_u32(bytes);
        let _min_y = loader.load_u32(bytes);
        let _min_z = loader.load_u32(bytes);
        let _max_x = loader.load_u32(bytes);
        let _max_y = loader.load_u32(bytes);
        let _max_z = loader.load_u32(bytes);
        
        let mut node = OctreeNode {
            children: [None, None, None, None, None, None, None, None],
            brick_index,
            is_leaf,
            bounds,
        };
        
        if !is_leaf {
            let child_mask = loader.load_u8(bytes);
            
            for i in 0..8 {
                if (child_mask & (1 << i)) != 0 {
                    let child_bounds = bounds.child_bounds(i);
                    let child_node = Self::deserialize_node(loader, bytes, child_bounds);
                    node.children[i] = Some(Box::new(child_node));
                }
            }
        }
        
        node
    }
}