## Lot cases.txt

| group | total | Pass | Fail | NotSupported | QualityWarning | CompatibilityWarning | Crash | Timeout | InternalError | NotRun | Other | pass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | 8186 | 5947 | 0 | 2239 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |
| api | 8186 | 5947 | 0 | 2239 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0 |

### Worst groups (Fail + Crash + Timeout + InternalError + other)

none

### Most frequent failure messages

| count | status | message | example |
|---:|---|---|---|

### Most frequent NotSupported reasons

| count | message | example |
|---:|---|---|
| 1472 | `VK_EXT_image_drm_format_modifier is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.get_physical_device_properties2.pnext_format_properties.r4g4_unorm_pack8.drm_format_mod_1 |
| 370 | `VK_KHR_fragment_shading_rate is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.unsupported_image_usage.optimal.fragment_shading_rate_attachment_undefined |
| 368 | `VK_EXT_multisampled_render_to_single_sampled is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.get_physical_device_properties2.pnext_format_properties.r4g4_unorm_pack8.subpass_resolve_query |
| 8 | `At least Vulkan #.# required to run test at vktApiFeatureInfo.cpp:#` | dEQP-VK.api.info.vulkan1p4.features |
| 3 | `Test is only for Android at vktApiFeatureInfo.cpp:#` | dEQP-VK.api.info.android.mandatory_extensions |
| 2 | `Profile not supported at vkProfileTests.inl:#` | dEQP-VK.api.info.profiles.roadmap_2022 |
| 1 | `Vulkan #.# is not supported at vkDeviceFeatures#.inl:#` | dEQP-VK.api.info.get_physical_device_properties2.features.create_device_with_promoted14_structures |
| 1 | `VK_KHR_shader_subgroup_rotate is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.get_physical_device_properties2.features.shader_subgroup_rotate_property_consistency_khr |
| 1 | `VK_EXT_discard_rectangles is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.ext_discard_rectangles |
| 1 | `VK_EXT_sample_locations is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.ext_sample_locations |
| 1 | `VK_EXT_external_memory_host is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.ext_external_memory_host |
| 1 | `VK_EXT_blend_operation_advanced is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.ext_blend_operation_advanced |
| 1 | `VK_EXT_conservative_rasterization is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.ext_conservative_rasterization |
| 1 | `VK_EXT_descriptor_indexing is not supported at vktApiFeatureInfo.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.ext_descriptor_indexing |
| 1 | `VK_KHR_vertex_attribute_divisor is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.khr_vertex_attribute_divisor |
| 1 | `VK_NV_mesh_shader is not supported at vktApiFeatureInfo.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.nv_mesh_shader |
| 1 | `VK_EXT_transform_feedback is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.ext_transform_feedback |
| 1 | `VK_EXT_fragment_density_map is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.fragment_density_map |
| 1 | `VK_NV_ray_tracing is not supported at vktApiFeatureInfo.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.nv_ray_tracing |
| 1 | `VK_EXT_line_rasterization is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.ext_line_rasterization |
| 1 | `VK_KHR_line_rasterization is not supported at vktTestCase.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.khr_line_rasterization |
| 1 | `VK_EXT_robustness# and VK_KHR_robustness# are not supported at vktApiFeatureInfo.cpp:#` | dEQP-VK.api.info.vulkan1p2_limits_validation.robustness2 |
