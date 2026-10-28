## Lot cases.txt

| group | total | Pass | Fail | NotSupported | QualityWarning | CompatibilityWarning | Crash | Timeout | InternalError | NotRun | Other | pass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | 949 | 125 | 45 | 775 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 71.8 |
| dEQP-GLES31 | 949 | 125 | 45 | 775 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 71.8 |

### Worst groups (Fail + Crash + Timeout + InternalError + other)

| group | total | Pass | Fail | NotSupported | QualityWarning | CompatibilityWarning | Crash | Timeout | InternalError | NotRun | Other | pass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dEQP-GLES31.functional | 949 | 125 | 45 | 775 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 71.8 |

### Most frequent failure messages

| count | status | message | example |
|---:|---|---|---|
| 19 | Fail | `glGetInternalFormativ() reported # supported sample counts at es#fInternalFormatQueryTests.cpp:#` | dEQP-GLES31.functional.state_query.internal_format.texture_2d_multisample.rgb10_a2ui_samples |
| 13 | Fail | `Got wrong error code` | dEQP-GLES31.functional.texture.multisample.negative.fbo_attach_different_fixed_state_tex_tex |
| 6 | Fail | `One or more iterations failed` | dEQP-GLES31.functional.state_query.texture_level.texture_2d_multisample.buffer_data_store_binding_integer |
| 2 | Fail | `Sample test failed` | dEQP-GLES31.functional.texture.multisample.samples_1.sample_mask_only |
| 2 | Fail | `Incorrect error was reported` | dEQP-GLES31.functional.debug.negative_coverage.get_error.buffer.blit_framebuffer_multisample |
| 1 | Fail | `Pixel comparison failed` | dEQP-GLES31.functional.texture.multisample.samples_1.sample_position |
| 1 | Fail | `Unused mask bits have effect` | dEQP-GLES31.functional.texture.multisample.samples_1.sample_mask_non_effective_bits |
| 1 | Fail | `getMultisamplefv: glGetError() returned GL_INVALID_VALUE at es#fMultisampleTests.cpp:#` | dEQP-GLES31.functional.multisample.default_framebuffer.sample_position |

### Most frequent NotSupported reasons

| count | message | example |
|---:|---|---|
| 195 | `Test requires GL_OES_shader_multisample_interpolation extension at es#fShaderMultisampleInterpolationTests.cpp:#` | dEQP-GLES31.functional.shaders.multisample_interpolation.sample_qualifier.default_framebuffer |
| 126 | `Test requires GL_OES_sample_variables extension or a context version #.# or higher. at es#fSampleVariableTests.cpp:#` | dEQP-GLES31.functional.shaders.sample_variables.num_samples.default_framebuffer |
| 63 | `Test requires GL_EXT_primitive_bounding_box extension` | dEQP-GLES31.functional.primitive_bounding_box.triangles.global_state.vertex_fragment.default_framebuffer_bbox_equal |
| 44 | `Test requires OES_texture_storage_multisample_#d_array extension` | dEQP-GLES31.functional.texture.multisample.samples_1.use_texture_color_2d_array |
| 44 | `Test requires GL_OES_texture_storage_multisample_#d_array extension` | dEQP-GLES31.functional.state_query.texture_level.texture_2d_multisample_array.samples_integer |
| 41 | `Test requires OES_texture_storage_multisample_#d_array extension or a context version equal or higher than #.# at es#fInternalFormatQueryTests.cpp:#` | dEQP-GLES31.functional.state_query.internal_format.texture_2d_multisample_array.r8_samples |
| 38 | `Test requires GL_OES_sample_shading extension or a context version #.# or higher. at es#fSampleVariableTests.cpp:#` | dEQP-GLES31.functional.shaders.sample_variables.sample_mask_in.bit_count_per_two_samples.default_framebuffer |
| 32 | `Requested sample count is greater than supported` | dEQP-GLES31.functional.texture.multisample.samples_1.use_texture_int_2d |
| 25 | `Requested sample count is greater than glGetInternalformativ(GL_SAMPLES) for this target/format` | dEQP-GLES31.functional.texture.multisample.samples_8.sample_mask_only |
| 20 | `Test requires line width #` | dEQP-GLES31.functional.primitive_bounding_box.wide_lines.global_state.vertex_fragment.default_framebuffer_bbox_equal |
| 19 | `Test requires extension: GL_OES_texture_storage_multisample_#d_array` | dEQP-GLES31.functional.state_query.texture.texture_2d_multisample_array.is_texture |
| 18 | `Test requires extension: GL_EXT_texture_border_clamp` | dEQP-GLES31.functional.state_query.texture.texture_2d_multisample.texture_swizzle_r_pure_int |
| 18 | `Test requires extension: GL_OES_texture_storage_multisample_#d_array, GL_EXT_texture_border_clamp` | dEQP-GLES31.functional.state_query.texture.texture_2d_multisample_array.texture_swizzle_r_pure_int |
| 18 | `Test requires GL_OES_sample_shading extension or a context version #.# or higher.` | dEQP-GLES31.functional.sample_shading.min_sample_shading.default_framebuffer_color |
| 12 | `Test requires GL_OES_shader_multisample_interpolation extension` | dEQP-GLES31.functional.state_query.multisample_interpolation.min_fragment_interpolation_offset.get_boolean |
| 11 | `Test requires GL_OES_shader_multisample_interpolation or GL_OES_sample_shading extension or a context version #.# or higher. at es#fSampleVariableTests.cpp:#` | dEQP-GLES31.functional.shaders.sample_variables.sample_pos.correctness.default_framebuffer |
| 10 | `Test requires GL_EXT_geometry_point_size extension` | dEQP-GLES31.functional.primitive_bounding_box.wide_points.global_state.vertex_geometry_fragment.default_framebuffer_bbox_equal |
| 8 | `Tests require GL_EXT_geometry_shader extension or higher context version. at es#fGeometryShaderTests.cpp:#` | dEQP-GLES31.functional.geometry_shading.layered.render_with_default_layer_2d_multisample_array |
| 7 | `The internal format is not supported in a context lower than #.# at es#fInternalFormatQueryTests.cpp:#` | dEQP-GLES31.functional.state_query.internal_format.texture_2d_multisample.r16f_samples |
| 7 | `Test requires GL_EXT_tessellation_point_size extension` | dEQP-GLES31.functional.primitive_bounding_box.wide_points.global_state.vertex_tessellation_fragment.default_framebuffer_bbox_equal |
| 6 | `Requested sample count is greater than GL_MAX_COLOR_TEXTURE_SAMPLES` | dEQP-GLES31.functional.texture.multisample.samples_8.sample_position |
| 5 | `Test requires larger GL_MAX_SAMPLE_MASK_WORDS` | dEQP-GLES31.functional.texture.multisample.samples_64.sample_mask_only |
| 2 | `Test requires GL_ANDROID_extension_pack_es#a extension` | dEQP-GLES31.functional.android_extension_pack.extensions.oes_shader_multisample_interpolation |
| 2 | `Test requires extension GL_ANDROID_extension_pack_es#a` | dEQP-GLES31.functional.android_extension_pack.shaders.es31.extension_directive.oes_shader_multisample_interpolation |
| 2 | `GLSL ES #.# is not supported at glsShaderLibraryCase.cpp:#` | dEQP-GLES31.functional.android_extension_pack.shaders.es32.extension_directive.oes_shader_multisample_interpolation |
| 1 | `Test requires GL_OES_texture_storage_multisample_#d_array` | dEQP-GLES31.functional.state_query.shader.sampler_type_multisample_array |
| 1 | `Test requires GL_OES_texture_stencil# extension or a context version equal or higher than #.# at es#fInternalFormatQueryTests.cpp:#` | dEQP-GLES31.functional.state_query.internal_format.texture_2d_multisample.stencil_index8_samples |
