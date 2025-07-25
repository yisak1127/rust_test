glslc shader/full_screen_triangle.vert -o shader/full_screen_triangle_vert.spv
glslc shader/svo_main.vert -o shader/svo_main_vert.spv
glslc shader/svo_main.frag -o shader/svo_main_frag.spv

glslc shader/main.vert -o shader/main_vert.spv
glslc shader/main.frag -o shader/main_frag.spv
glslc shader/main_frontface.vert -o shader/main_frontface_vert.spv
glslc shader/simple.frag -o shader/simple_frag.spv

glslc shader/vbuffer.vert -o shader/vbuffer_vert.spv
glslc shader/vbuffer_nonindexed.vert -o shader/vbuffer_nonindexed_vert.spv
glslc shader/vbuffer_leadingvertex.vert -o shader/vbuffer_leadingvertex_vert.spv
glslc shader/vbuffer_getattributeatvertex.vert -o shader/vbuffer_getattributeatvertex_vert.spv

glslc shader/vbuffer_color.frag -o shader/vbuffer_color_frag.spv
glslc shader/vbuffer_primid.frag -o shader/vbuffer_primid_frag.spv
glslc shader/vbuffer_nonindexed.frag -o shader/vbuffer_nonindexed_frag.spv
glslc shader/vbuffer_leadingvertex.frag -o shader/vbuffer_leadingvertex_frag.spv
glslc shader/vbuffer_getattributeatvertex.frag -o shader/vbuffer_getattributeatvertex_frag.spv

glslc shader/depth_pyramid_first_mip.comp -o shader/depth_pyramid_first_mip.spv
glslc shader/depth_pyramid_downsample.comp -o shader/depth_pyramid_downsample.spv
glslc shader/depth_pyramid_downsample_all.comp -o shader/depth_pyramid_downsample_all.spv

glslc shader/culling.comp -o shader/culling.spv
glslc shader/culling_debug.frag -o shader/culling_debug_frag.spv

