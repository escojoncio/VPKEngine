## Lot cases.txt

| group | total | Pass | Fail | NotSupported | QualityWarning | CompatibilityWarning | Crash | Timeout | InternalError | NotRun | Other | pass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | 3321 | 3117 | 27 | 176 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 99.1 |
| dEQP-GLES3 | 3321 | 3117 | 27 | 176 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 99.1 |

### Worst groups (Fail + Crash + Timeout + InternalError + other)

| group | total | Pass | Fail | NotSupported | QualityWarning | CompatibilityWarning | Crash | Timeout | InternalError | NotRun | Other | pass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dEQP-GLES3.functional | 3321 | 3117 | 27 | 176 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 99.1 |

### Most frequent failure messages

| count | status | message | example |
|---:|---|---|---|
| 15 | Fail | `Image comparison failed` | dEQP-GLES3.functional.fbo.blit.rect.out_of_bounds_linear |
| 10 | Fail | `Framebuffer checked as complete, expected incomplete` | dEQP-GLES3.functional.fbo.completeness.renderable.renderbuffer.color0.red_unsigned_byte |
| 1 | Fail | `Error code mismatch` | dEQP-GLES3.functional.fbo.api.attach_names |
| 1 | Fail | `Got invalid error` | dEQP-GLES3.functional.negative_api.buffer.blit_framebuffer_multisample |

### Most frequent NotSupported reasons

| count | message | example |
|---:|---|---|
| 122 | `Not supported in MSAA config` | dEQP-GLES3.functional.fbo.blit.default_framebuffer.rgba8 |
| 22 | `Sample count not supported` | dEQP-GLES3.functional.fbo.msaa.8_samples.rgba8 |
| 16 | `Maximum sample count returned by glGetInternalformativ() for GL_RGBA# is only #` | dEQP-GLES3.functional.multisample.fbo_8_samples.num_samples_polygon |
| 12 | `Separate depth and stencil buffers not supported` | dEQP-GLES3.functional.fbo.blit.depth_stencil.depth_component32f_stencil_index8_basic |
| 4 | `wide line support required` | dEQP-GLES3.functional.rasterization.fbo.rbo_multisample_4.primitives.lines_wide |
