## GLES CTS rapida on vpgles (macOS, vulkan-cts-1.4.6.2)

| group | total | Pass | Fail | NotSupported | QualityWarning | CompatibilityWarning | Crash | Timeout | InternalError | NotRun | Other | pass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | 7634 | 7189 | 64 | 380 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 99.1 |
| dEQP-GLES2 | 2191 | 2149 | 6 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99.7 |
| dEQP-GLES3 | 4243 | 4188 | 30 | 24 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 99.3 |
| dEQP-GLES31 | 1200 | 852 | 28 | 320 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 96.8 |

### Worst groups (Fail + Crash + Timeout + InternalError + other)

| group | total | Pass | Fail | NotSupported | QualityWarning | CompatibilityWarning | Crash | Timeout | InternalError | NotRun | Other | pass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dEQP-GLES3.functional | 4237 | 4182 | 30 | 24 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 99.3 |
| dEQP-GLES31.functional | 1194 | 846 | 28 | 320 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 96.8 |
| dEQP-GLES2.functional | 2185 | 2143 | 6 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99.7 |

### Most frequent failure messages

| count | status | message | example |
|---:|---|---|---|
| 20 | Fail | `Image comparison failed` | dEQP-GLES3.functional.fbo.blit.default_framebuffer.rg8_linear_out_of_bounds_blit_to_default |
| 9 | Fail | `Image verification failed` | dEQP-GLES2.functional.texture.mipmap.2d.projected.linear_linear_clamp |
| 7 | Fail | `could not find target resource` | dEQP-GLES31.functional.program_interface_query.program_input.type.separable_fragment.array.vec4 |
| 5 | Fail | `One or more iterations failed` | dEQP-GLES31.functional.state_query.texture_level.texture_2d_multisample.buffer_offset_float |
| 3 | Fail | `Expected #(#x#) , got #(#x#)` | dEQP-GLES3.functional.state_query.texture.texture_2d_array_texture_max_level_gettexparameteri |
| 3 | Fail | `glGetInternalFormativ() reported # supported sample counts at es#fInternalFormatQueryTests.cpp:#` | dEQP-GLES31.functional.state_query.internal_format.renderbuffer.rg16i_samples |
| 2 | Fail | `Detected variance between two invariant values` | dEQP-GLES2.functional.shaders.invariance.mediump.loop_1 |
| 2 | Fail | `expected shaders to compile and link properly, but failed to link.` | dEQP-GLES3.functional.shaders.linkage.varying.struct.ivec3 |
| 2 | Fail | `Invalid std# layout` | dEQP-GLES3.functional.ubo.random.nested_structs_arrays.2 |
| 2 | Fail | `Got invalid error` | dEQP-GLES3.functional.negative_api.buffer.copy_buffer_sub_data |
| 2 | Fail | `Expected #, got #` | dEQP-GLES3.functional.state_query.texture.texture_3d_texture_base_level_gettexparameterf |
| 1 | Fail | `Got invalid integer value` | dEQP-GLES2.functional.state_query.floats.sample_coverage_value_getinteger |
| 1 | Fail | `GetProgramResourceIndex returned unexpected values` | dEQP-GLES31.functional.program_interface_query.program_output.resource_list.separable_vertex.var_struct |
| 1 | Fail | `Invalid std# or std# layout` | dEQP-GLES31.functional.ssbo.layout.random.nested_structs_arrays_instance_arrays.4 |
| 1 | Fail | `Expected GL_TRUE, got GL_FALSE` | dEQP-GLES31.functional.state_query.vertex_attribute_binding.indexed.vertex_binding_buffer_getboolean |
| 1 | Fail | `Result verification failed` | dEQP-GLES31.functional.texture.gather.basic.2d_array.rgba8i.base_level.level_1 |
| 1 | Fail | `Sample test failed` | dEQP-GLES31.functional.texture.multisample.samples_1.sample_mask_only |
| 1 | Fail | `Incorrect uniform location` | dEQP-GLES31.functional.uniform_location.nested_array.usampler2D_fragment |

### Most frequent NotSupported reasons

| count | message | example |
|---:|---|---|
| 38 | `GL_EXT_gpu_shader# required` | dEQP-GLES31.functional.texture.gather.offset_dynamic.implementation_offset.2d.depth32f.size_npot.compare_greater.mirrored_repeat_clamp_to_edge |
| 28 | `GLSL ES #.# is not supported at glsShaderLibraryCase.cpp:#` | dEQP-GLES31.functional.separate_shader.validation.es32.io_blocks.mismatch_member_precision |
| 26 | `Extension not supported: &apos;&apos; at es#fTextureSpecificationTests.cpp:#` | dEQP-GLES31.functional.texture.specification.basic_teximage3d.r8_snorm_cube_array |
| 22 | `Test requires extension: GL_EXT_texture_border_clamp` | dEQP-GLES31.functional.state_query.sampler.set_pure_uint.texture_min_filter |
| 20 | `Not supported` | dEQP-GLES2.functional.fbo.render.color_clear.rbo_bgra_depth_component16_stencil_index8 |
| 20 | `Test requires OES_shader_image_atomic extension` | dEQP-GLES31.functional.image_load_store.2d.atomic.exchange_r32i_result |
| 18 | `Test requires GL_EXT_tessellation_shader extension` | dEQP-GLES31.functional.program_interface_query.buffer_variable.referenced_by.separable_tess_ctrl.named_block.float |
| 16 | `GL_EXT_texture_cube_map_array not supported` | dEQP-GLES31.functional.texture.filtering.cube_array.combinations.linear_mipmap_linear_linear_repeat_mirror |
| 11 | `Test requires extension GL_EXT_texture_cube_map_array or a context version equal or higher than #.# at es#fFboColorbufferTests.cpp:#` | dEQP-GLES31.functional.fbo.color.texcubearray.r11f_g11f_b10f |
| 10 | `Implementation limits execeeded: &apos;&apos; at es#fTransformFeedbackTests.cpp:#` | dEQP-GLES3.functional.transform_feedback.array.separate.lines.lowp_mat2x3 |
| 10 | `Test requires GL_OES_shader_multisample_interpolation extension at es#fShaderMultisampleInterpolationTests.cpp:#` | dEQP-GLES31.functional.shaders.multisample_interpolation.interpolate_at_offset.at_sample_position.multisample_rbo_2 |
| 9 | `Multiple attachments not supported: &apos;m_ctx.haveMultiColorAtts()&apos; at glsFboCompletenessTests.cpp:#` | dEQP-GLES2.functional.fbo.completeness.attachment_combinations.none_rbo_none_tex |
| 9 | `GL_EXT_gpu_shader# extension is required for dynamic indexing of sampler arrays.: &apos;m_context.getContextInfo().isExtensionSupported(&quot;GL_EXT_gpu_shader#` | dEQP-GLES31.functional.shaders.opaque_type_indexing.sampler.dynamically_uniform.compute.sampler2d |
| 8 | `Test requires GL_OES_sample_variables extension or a context version #.# or higher. at es#fSampleVariableTests.cpp:#` | dEQP-GLES31.functional.shaders.sample_variables.max_samples.multisample_rbo_8 |
| 8 | `Test requires OES_texture_storage_multisample_#d_array extension or a context version equal or higher than #.# at es#fInternalFormatQueryTests.cpp:#` | dEQP-GLES31.functional.state_query.internal_format.texture_2d_multisample_array.r11f_g11f_b10f_samples |
| 8 | `texture srgb rg# not supported` | dEQP-GLES31.functional.texture.format.sized.cube_array.depth_component16_pot |
| 6 | `TEXTURE_#D target not supported. at es#fCopyImageTests.cpp:#` | dEQP-GLES31.functional.copy_image.compressed.viewclass_astc_12x10_rgba.rgba_astc_12x10_khr_srgb8_alpha8_astc_12x10_khr.cubemap_to_texture3d |
| 6 | `GL_EXT_shader_framebuffer_fetch not supported` | dEQP-GLES31.functional.shaders.framebuffer_fetch.basic.framebuffer_texture_level |
| 6 | `Test requires GL_OES_texture_storage_multisample_#d_array extension` | dEQP-GLES31.functional.state_query.texture_level.texture_2d_multisample_array.blue_size_float |
| 6 | `Test requires GL_EXT_texture_cube_map_array extension` | dEQP-GLES31.functional.state_query.texture_level.texture_cube_array.alpha_size_float |
| 5 | `Sample count not supported` | dEQP-GLES3.functional.fbo.msaa.8_samples.depth24_stencil8 |
| 5 | `OpenGL ES #.#, GL_EXT_gpu_shader# not supported and OpenGL #.#` | dEQP-GLES31.functional.shaders.builtin_functions.common.fma.float_highp_compute |
| 4 | `No suitable compressed formats found` | dEQP-GLES2.functional.negative_api.texture.compressedtexsubimage2d_level_max_cube |
| 4 | `Test requires GL_EXT_shader_io_blocks extension` | dEQP-GLES31.functional.program_interface_query.program_input.name_length.interface_blocks.in.named_block.var_array |
| 4 | `Test not supported in this context version.: &apos;contextSupports(m_context.getRenderContext().getType(), m_minimumVersion)&apos; at es#fIntegerStateQueryTests` | dEQP-GLES31.functional.state_query.integer.debug_next_logged_message_length_getinteger64 |
| 4 | `Test requires OES_texture_storage_multisample_#d_array extension` | dEQP-GLES31.functional.state_query.integer.texture_binding_2d_multisample_array_getfloat |
| 4 | `Test requires extension: GL_EXT_texture_cube_map_array` | dEQP-GLES31.functional.state_query.texture.texture_cube_array.texture_compare_func_integer |
| 3 | `No multisample buffers` | dEQP-GLES2.functional.multisample.constancy_sample_coverage |
| 3 | `GL_EXT_texture_sRGB_RG# is not supported. at es#fTextureFilteringTests.cpp:#` | dEQP-GLES3.functional.texture.filtering.3d.formats.srgb_rg8_linear_mipmap_nearest |
| 3 | `GL_MAX_SHADER_STORAGE_BLOCK_SIZE is too small for vertex attrib buffers` | dEQP-GLES31.functional.draw_indirect.compute_interop.large.drawarrays_separate_grid_1000x1000_drawcount_5000 |
| 3 | `Vertex shader: not enough uniforms available for test at es#fLayoutBindingTests.cpp:#` | dEQP-GLES31.functional.layout_binding.negative.image.image2d.binding_contradictory |
| 3 | `Not enough uniforms available for test at es#fLayoutBindingTests.cpp:#` | dEQP-GLES31.functional.layout_binding.negative.image.image2d.vertex_binding_over_max_array |
| 3 | `GL_EXT_geometry_shader extension is required to run geometry shader tests.: &apos;m_context.getContextInfo().isExtensionSupported(&quot;GL_EXT_geometry_shader&q` | dEQP-GLES31.functional.shaders.opaque_type_indexing.sampler.const_literal.geometry.isamplercubearray |
| 3 | `Test requires extension: GL_EXT_texture_cube_map_array, GL_EXT_texture_border_clamp` | dEQP-GLES31.functional.state_query.texture.texture_cube_array.texture_compare_func_pure_uint |
| 3 | `GL_STENCIL_INDEX# requires GL_OES_texture_stencil#` | dEQP-GLES31.functional.stencil_texturing.format.stencil_index8_2d |
| 3 | `Requested sample count is greater than supported` | dEQP-GLES31.functional.texture.multisample.samples_10.use_texture_depth_2d |
| 2 | `Separate depth and stencil buffers not supported` | dEQP-GLES3.functional.fbo.blit.depth_stencil.depth_component24_stencil_index8_depth_only |
| 2 | `GL_EXT_texture_sRGB_R# is not supported. at es#fTextureFilteringTests.cpp:#` | dEQP-GLES3.functional.texture.filtering.2d.formats.srgb_r8_linear_mipmap_linear |
| 2 | `Test requires OES_shader_image_atomic extension at es#fBasicComputeShaderTests.cpp:#` | dEQP-GLES31.functional.compute.basic.image_atomic_op_local_size_1 |
| 2 | `Not enough uniforms available for test` | dEQP-GLES31.functional.layout_binding.image.image2d.vertex_binding_max_array |
