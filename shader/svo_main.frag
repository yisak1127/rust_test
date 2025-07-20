#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (binding = 0) uniform UBO {
    mat4 world_to_screen;
    vec4 color;
    vec4 camera_position;
    vec4 volume_scale;
    vec4 center_to_edge;
    vec4 texel_scale;
    uint brick_size;
} ubo;

struct InstanceData
{
    vec4 position;
    uint brick_index;
    uint brick_size;
    uint padding[2];
};

struct VisibilityData
{
    uint index;
};

struct OctreeNode
{
    uvec3 bounds_min;
    uvec3 bounds_max;
    uint brick_index;
    uint child_mask;
    uint children_offset;
    uint is_leaf;
    uint padding[2];
};

layout(std430, binding = 1) buffer Instances
{
    InstanceData instances[];
};

layout(std430, binding = 2) buffer Visibility
{
    VisibilityData visibility[];
};

layout (binding = 3) uniform sampler3D samplerBricks;

layout(std430, binding = 4) buffer OctreeBuffer
{
    OctreeNode octree_nodes[];
};

layout (location = 0) in vec3 o_uvw;
layout (location = 1) in vec4 o_local_camera_pos_lod;
layout (location = 2) in vec3 o_local_pos;
layout (location = 3) in flat uint o_brick_index;

layout (location = 0) out vec4 uFragColor;

bool outside(vec3 uwv) {
    return any(greaterThan(abs(uwv - vec3(0.5, 0.5, 0.5)), vec3(0.5, 0.5, 0.5)));
}

vec3 normal(vec3 uvw) {
    float lod = o_local_camera_pos_lod.w;
    vec3 e = ubo.texel_scale.xyz * 0.5;
    float xm = textureLod(samplerBricks, uvw + vec3(-e.x, 0,    0), lod).x;
    float xp = textureLod(samplerBricks, uvw + vec3( e.x, 0,    0), lod).x;
    float ym = textureLod(samplerBricks, uvw + vec3( 0,   -e.y, 0), lod).x;
    float yp = textureLod(samplerBricks, uvw + vec3( 0,   e.y,  0), lod).x;
    float zm = textureLod(samplerBricks, uvw + vec3( 0,   0, -e.z), lod).x;
    float zp = textureLod(samplerBricks, uvw + vec3( 0,   0,  e.z), lod).x;
    return normalize(vec3(xp - xm, yp - ym, zp - zm));
}

void main() {
    vec3 ray_pos = o_uvw;
    vec3 ray_dir = normalize(o_local_pos - o_local_camera_pos_lod.xyz);

    ray_dir *= ubo.volume_scale.xyz;

    // Sample from the brick texture using the brick index
    // For now, we'll use a simple approach - in a full implementation,
    // you'd calculate the proper texture coordinates based on brick layout
    float s = textureLod(samplerBricks, ray_pos, o_local_camera_pos_lod.w).x;
    s = s * 2.0 - 1.0;

    float d = s;
    if (s > 0.00025) 
    {
        for (uint i=0; i<256; ++i) {  // Reduced iterations for SVO
            vec3 uvw = ray_pos + ray_dir * d;
            if (outside(uvw)) {
                discard;
                break;
            }
            float s = textureLod(samplerBricks, uvw, o_local_camera_pos_lod.w).x;
            s = s * 2.0 - 1.0;
            d += s;
            if (s < 0.00025) break;
        }
    }
    
    vec3 final_normal = normal(ray_pos + ray_dir * d);
    
    // Color based on brick index for debugging
    vec3 brick_color = vec3(
        float((o_brick_index * 73) % 255) / 255.0,
        float((o_brick_index * 151) % 255) / 255.0,
        float((o_brick_index * 211) % 255) / 255.0
    );
    
    uFragColor = vec4(final_normal * 0.7 + brick_color * 0.3, 1.0);
}