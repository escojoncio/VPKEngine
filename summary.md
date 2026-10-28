## Vulkan CTS rapida on VPVulkan (macOS, vulkan-cts-1.4.6.2)

| group | total | Pass | Fail | NotSupported | QualityWarning | CompatibilityWarning | Crash | Timeout | InternalError | NotRun | Other | pass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | 36087 | 11602 | 26 | 24445 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 99.7 |
| api | 7964 | 3396 | 0 | 4555 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 99.6 |
| binding_model | 2054 | 458 | 0 | 1596 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| clipping | 163 | 40 | 0 | 123 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| compute | 1038 | 33 | 0 | 1005 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| depth | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| draw | 2100 | 566 | 0 | 1534 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| dynamic_state | 163 | 24 | 0 | 139 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| fragment_operations | 33 | 17 | 0 | 15 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 94.4 |
| geometry | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| glsl | 2277 | 1387 | 0 | 890 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| image | 1475 | 426 | 0 | 1049 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| info | 21 | 19 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| memory | 1530 | 651 | 0 | 879 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| multiview | 694 | 569 | 2 | 123 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99.6 |
| pipeline | 3067 | 707 | 11 | 2349 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 98.5 |
| query_pool | 869 | 23 | 0 | 846 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| rasterization | 1263 | 16 | 1 | 1246 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 94.1 |
| renderpasses | 2182 | 1158 | 5 | 1019 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99.6 |
| robustness | 792 | 31 | 0 | 761 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| spirv_assembly | 1121 | 196 | 1 | 924 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99.5 |
| ssbo | 561 | 550 | 0 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| subgroups | 1233 | 131 | 0 | 1102 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| synchronization | 1338 | 128 | 0 | 1210 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| synchronization2 | 1018 | 119 | 1 | 898 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99.2 |
| tessellation | 190 | 0 | 0 | 190 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| texture | 2275 | 571 | 5 | 1699 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99.1 |
| ubo | 625 | 386 | 0 | 239 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |

### Worst groups (Fail + Crash + Timeout + InternalError + other)

| group | total | Pass | Fail | NotSupported | QualityWarning | CompatibilityWarning | Crash | Timeout | InternalError | NotRun | Other | pass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pipeline.monolithic | 3067 | 707 | 11 | 2349 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 98.5 |
| texture.mipmap | 47 | 17 | 5 | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 77.3 |
| renderpasses.renderpass2 | 1339 | 487 | 4 | 848 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99.2 |
| multiview.input_attachments | 8 | 7 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 87.5 |
| multiview.renderpass2 | 234 | 192 | 1 | 41 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99.5 |
| renderpasses.renderpass1 | 843 | 671 | 1 | 171 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99.9 |
| rasterization.culling | 7 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 85.7 |
| spirv_assembly.instruction | 671 | 131 | 1 | 539 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99.2 |
| synchronization2.signal_order | 93 | 8 | 1 | 84 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 88.9 |

### Most frequent failure messages

| count | status | message | example |
|---:|---|---|---|
| 5 | Fail | `fail` | dEQP-VK.texture.mipmap.2d.projected.nearest_linear_mirror |
| 4 | Fail | `Image mismatch` | dEQP-VK.pipeline.monolithic.sampler.separate_stencil_usage.view_type.2d.format.d24_unorm_s8_uint.address_modes.uvw_mode_mirror_clamp_to_edge_mode_repeat_mode_clamp_to_edge |
| 4 | Fail | `Result verification failed` | dEQP-VK.renderpasses.renderpass1.dedicated_allocation.attachment_allocation.input_output.24 |
| 3 | Fail | `Fail` | dEQP-VK.multiview.input_attachments.no_queries.8_1_1_8 |
| 3 | Fail | `Incorrect multisampled rendering for depth attachment, stencil attachment` | dEQP-VK.pipeline.monolithic.multisample.misc.clear_attachments.r8g8b8a8_unorm_r16g16b16a16_sfloat_r32g32b32a32_uint_d32_sfloat_s8_uint.2x.ds_resolve_sample_zero.sub_framebuffer |
| 2 | Fail | `Ref:(#, #, #, #) Threshold:(#, #, #, #) Color:(#, #, #, #) at vktPipelineSamplerBorderSwizzleTests.cpp:#` | dEQP-VK.pipeline.monolithic.sampler.border_swizzle.d32_sfloat.rg1a.opaque_white.gather_1.no_swizzle_hint |
| 1 | Fail | `Incorrect multisampled rendering for depth attachment` | dEQP-VK.pipeline.monolithic.multisample.misc.dynamic_rendering.clear_attachments.r8g8b8a8_unorm_r16g16b16a16_sfloat_r16g16b16a16_sint_d16_unorm.4x.ds_resolve_sample_zero.whole_framebuffer |
| 1 | Fail | `Incorrect multisampled rendering for color attachment #, color attachment #, color attachment #, stencil attachment` | dEQP-VK.pipeline.monolithic.multisample.misc.dynamic_rendering.multi_renderpass.r8g8b8a8_unorm_r16g16b16a16_sfloat_r16g16b16a16_sint_s8_uint.random_221 |
| 1 | Fail | `Incorrect rendering` | dEQP-VK.rasterization.culling.front_triangles_reverse_point |
| 1 | Fail | `Resource returned doesn&apos;t match with expected` | dEQP-VK.spirv_assembly.instruction.graphics.float16.opcompositeinsert.v3f16_vert |
| 1 | Fail | `Memory contents don&apos;t match` | dEQP-VK.synchronization2.signal_order.timeline_semaphore.write_image_vertex_read_image_vertex.image_64x64x8_r32_sfloat |

### Most frequent NotSupported reasons

| count | message | example |
|---:|---|---|
| 1954 | `VK_KHR_maintenance# is not supported at vktTestCase.cpp:#` | dEQP-VK.api.buffer_view.access.uniform_texel_buffer.a1b5g5r5_unorm_pack16 |
| 901 | `Requested core feature is not supported: tessellationShader at vktTestCase.cpp:#` | dEQP-VK.pipeline.monolithic.cache.merge.vertex_stage_tessellation_control_stage_tessellation_evaluation_stage_fragment_stage.src_empty_from_data_dst_empty |
| 838 | `Pipeline statistics queries are not supported at vktQueryPoolStatisticsTests.cpp:#` | dEQP-VK.query_pool.statistics_query.clipping_invocations.primary.32bits_dstoffset_cmdcopyquerypoolresults_point_list_with_no_color_attachments_vertex |
| 784 | `VkPhysicalDeviceCooperativeMatrixFeaturesKHR::cooperativeMatrix not supported at vktComputeCooperativeMatrixTests.cpp:#` | dEQP-VK.compute.pipeline.cooperative_matrix.khr_a.subgroupscope.add.uint8_uint8.physical_buffer.rowmajor.tensorlayout |
| 754 | `Exclusive compute queue not supported. at vktTextureTestUtil.cpp:#` | dEQP-VK.texture.filtering.2d.combinations.linear.linear.clamp_to_border.clamp_to_border_compute |
| 701 | `Requested core feature is not supported: geometryShader at vktTestCase.cpp:#` | dEQP-VK.multiview.dynamic_rendering.index.geometry_shader.no_queries.15 |
| 682 | `Sparse bindings feature is not supported at vktApiBufferTests.cpp:#` | dEQP-VK.api.buffer.basic.max_size_sparse |
| 671 | `Exclusive compute queue not supported. at vktTestCase.cpp:#` | dEQP-VK.api.command_buffers.indirect_compute_dispatch_offsets_0_12 |
| 639 | `VK_EXT_depth_bias_control is not supported at vktTestCase.cpp:#` | dEQP-VK.rasterization.depth_bias_control.d16_unorm.float_exact.constant.constant_depth_0_125.target_bias_0_0625.static_no_clamp_secondary_cmd_buffer_unspecified_fb |
| 593 | `Source format not supported at vktApiBlittingTests.cpp:#` | dEQP-VK.api.copy_and_blit.copy_commands2.blit_image.all_formats.color.1d.a2r10g10b10_sint_pack32.a2r10g10b10_sint_pack32.general_optimal_nearest |
| 573 | `VK_KHR_map_memory# is not supported at vktTestCase.cpp:#` | dEQP-VK.memory.mapping.dedicated_alloc.buffer.full.1048577.flush_map2 |
| 520 | `VK_KHR_external_semaphore_fd is not supported at vktTestCase.cpp:#` | dEQP-VK.synchronization.cross_instance.dedicated.write_blit_image_read_blit_image.image_128x128_r16g16b16a16_uint_binary_semaphore_fence_fd |
| 503 | `VK_KHR_external_semaphore_win# is not supported at vktTestCase.cpp:#` | dEQP-VK.synchronization.cross_instance.dedicated.write_blit_image_read_image_compute.image_128_r32_uint_binary_semaphore_win32_kmt |
| 497 | `VK_EXT_image_drm_format_modifier is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.get_physical_device_properties2.pnext_format_properties.a1r5g5b5_unorm_pack16.drm_format_mod_1 |
| 493 | `VK_EXT_shader_tile_image is not supported at vktTestCase.cpp:#` | dEQP-VK.rasterization.shader_tile_image.coherent.color.samples_1.multi_draws.multi_patches.a2b10g10r10_uint_pack32 |
| 448 | `Tessellation shader not supported` | dEQP-VK.clipping.user_defined.clip_cull_distance.vert_tess.1_7 |
| 368 | `No matching queue found: findQueueFamilyIndexWithCaps(requiredCaps=#x#, excludedCaps=#x#) at vktTestCase.cpp:#` | dEQP-VK.api.buffer_marker.compute.default_mem.bottom_of_pipe.memory_dep.dispatch_offset_24 |
| 323 | `Device does not support subgroup partitioned operations at vktSubgroupsPartitionedTests.cpp:#` | dEQP-VK.subgroups.partitioned.compute.subgroupadd_int64_t |
| 318 | `Destination format not supported at vktApiBlittingTests.cpp:#` | dEQP-VK.api.copy_and_blit.copy_commands2.blit_image.all_formats.color.2d.astc_10x10_unorm_block.r4g4_unorm_pack8.optimal_general_linear |
| 291 | `# is not supported at vktBindingShaderAccessTests.cpp:#` | dEQP-VK.binding_model.shader_access.primary_cmd_buf.bind.combined_image_sampler_immutable.geometry.multiple_descriptor_sets.single_descriptor.cube_array_base_slice |
| 274 | `Stencil resolve mode not supported at vktRenderPassDepthStencilResolveTests.cpp:#` | dEQP-VK.renderpasses.renderpass2.depth_stencil_resolve.image_2d_17_1.samples_16.d16_unorm_s8_uint.depth_max_stencil_max_testing_depth |
| 267 | `Concurrent requires more than # queue family at vktSynchronizationOperationMultiQueueTests.cpp:#` | dEQP-VK.synchronization.op.multi_queue.binary_semaphore.write_blit_image_read_image_tess_eval.image_128x128_r16g16b16a16_uint_concurrent_maintenance9 |
| 235 | `VK_KHR_acceleration_structure is not supported at vktTestCase.cpp:#` | dEQP-VK.binding_model.descriptor_update.acceleration_structure.ray_query.regular.frag |
| 226 | `Tessellation not supported at vktSpvAsmGraphicsShaderTestUtil.cpp:#` | dEQP-VK.spirv_assembly.instruction.graphics.16bit_storage.input_output_float_16_to_16.scalar0_tesse |
| 219 | `Format not supported at vktApiCopyImageToImageTests.cpp:#` | dEQP-VK.api.copy_and_blit.copy_commands2.image_to_image.all_formats.color.2d_to_2d.a2r10g10b10_snorm_pack32.r32_sfloat.optimal_optimal |
| 218 | `VK_KHR_ray_tracing_pipeline is not supported at vktTestCase.cpp:#` | dEQP-VK.pipeline.monolithic.bind_point.graphics_raytracing.write_template_push.setup_rs_gs_rp_gp.cmd_tracerays_draw |
| 215 | `Operation not supported with this image format at vktImageCompressionTranscodingSupport.cpp:#` | dEQP-VK.image.texel_view_compatible.compute.basic.1d_image.image_load.bc1_rgba_unorm_block.r32g32_uint |
| 208 | `VK_EXT_pipeline_protected_access is not supported at vktTestCase.cpp:#` | dEQP-VK.pipeline.monolithic.image.dedicated_allocation.sampling_type.combined.view_type.1d_array.format.r8g8b8a8_unorm.count_1.size.pipeline_protected_flag_2x1_array_of_6 |
| 205 | `Format not supported for transfer at vktApiImageClearingTests.cpp:#` | dEQP-VK.api.image_clearing.core.clear_color_attachment.cube_layers.r16_sscaled |
| 201 | `robustBufferAccess# not supported at vktRobustnessExtsTests.cpp:#` | dEQP-VK.robustness.pipeline_robustness.robustness2.bind.notemplate.rgba32f.dontunroll.nonvolatile.storage_buffer.readwrite.no_fmt_qual.len_256.samples_1.1d.comp_compute |
| 194 | `VK_FUCHSIA_external_semaphore is not supported at vktTestCase.cpp:#` | dEQP-VK.synchronization.cross_instance.dedicated.write_blit_image_read_image_compute_indirect.image_128x128_r8g8b8a8_unorm_binary_semaphore_zircon_handle |
| 175 | `VK_KHR_workgroup_memory_explicit_layout is not supported at vktTestCase.cpp:#` | dEQP-VK.compute.pipeline.workgroup_memory_explicit_layout.alias.f16_to_u16_scalar_func_write |
| 174 | `Format not supported at vktApiBufferViewCreateTests.cpp:#` | dEQP-VK.api.buffer_view.create.dedicated_alloc.storage.a2r10g10b10_sint_pack32 |
| 165 | `Format does not support shadow sampling at vktTextureShadowTests.cpp:#` | dEQP-VK.texture.shadow.1d.linear.greater_r32_sfloat |
| 163 | `VK_KHR_fragment_shading_rate is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.unsupported_image_usage.linear.fragment_shading_rate_attachment_a1r5g5b5_unorm_pack16 |
| 158 | `VK_EXT_multisampled_render_to_single_sampled is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.get_physical_device_properties2.pnext_format_properties.a1r5g5b5_unorm_pack16.subpass_resolve_query |
| 157 | `Dynamic indexing not supported at vktBindingDescriptorSetRandomTests.cpp:#` | dEQP-VK.binding_model.descriptorset_random.sets16.dynindexed.noubo.nosbo.nosampledimg.storageimghigh.iublimithigh.uab.vert.noia.0 |
| 153 | `longVector not supported at vktSubgroupsTestsUtils.cpp:#` | dEQP-VK.subgroups.arithmetic.compute.subgroupadd_vec8_int |
| 152 | `robustImageAccess# not supported at vktRobustnessExtsTests.cpp:#` | dEQP-VK.robustness.robustness2.bind.notemplate.r32f.dontunroll.volatile.storage_image.fmt_qual.img.samples_1.1d.comp_compute |
| 148 | `Required feature: shaderResourceResidency. at vktShaderRender.cpp:#` | dEQP-VK.glsl.texture_functions.texelfetch.sparse_isampler2darray_vertex |
