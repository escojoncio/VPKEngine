// OpenGL ES 3.1 / 3.2 and KHR_debug / EXT_disjoint_timer_query numbers used by the 3.1 layer of
// vpgles (values from the Khronos specifications). Kept apart from gl_enums.h, which holds the 2.0 / 3.0
// set the rest of the layer was written against.
#pragma once

#include "gl_enums.h"

namespace vpgles {
namespace gl {

// ---- advertised level -------------------------------------------------------------------------------
// The one place the version vpgles tells the guest is chosen. 3.1 is what is implemented end to end
// (vertex attrib binding, indirect draws, copy image, program interfaces, texture buffers, compute with
// the GLSL ES 3.10 compiler: SSBO, images, atomic counters; checked by gles_tests es31); program
// pipelines (glUseProgramStages) are the one ES 3.1 feature still missing; 3.2 adds geometry / tessellation / advanced blend / sample shading,
// which are not implemented, so it is not announced. The coordinator edits only these two numbers.
constexpr int kAdvertisedMajor = 3;
constexpr int kAdvertisedMinor = 1;

// ---- queries -----------------------------------------------------------------------------------------
constexpr GLenum TIMESTAMP_EXT = 0x8E28,
                 QUERY_COUNTER_BITS_EXT = 0x8864;

// ---- vertex attrib binding ---------------------------------------------------------------------------
constexpr GLenum VERTEX_ATTRIB_BINDING = 0x82D4, VERTEX_ATTRIB_RELATIVE_OFFSET = 0x82D5, VERTEX_BINDING_DIVISOR = 0x82D6,
                 VERTEX_BINDING_OFFSET = 0x82D7, VERTEX_BINDING_STRIDE = 0x82D8, MAX_VERTEX_ATTRIB_RELATIVE_OFFSET = 0x82D9,
                 VERTEX_BINDING_BUFFER = 0x8F4F;

// ---- textures ----------------------------------------------------------------------------------------
constexpr GLenum TEXTURE_WIDTH = 0x1000, TEXTURE_HEIGHT = 0x1001, TEXTURE_INTERNAL_FORMAT = 0x1003,
                 TEXTURE_RED_SIZE = 0x805C, TEXTURE_GREEN_SIZE = 0x805D, TEXTURE_BLUE_SIZE = 0x805E,
                 TEXTURE_ALPHA_SIZE = 0x805F, TEXTURE_DEPTH = 0x8071, TEXTURE_DEPTH_SIZE = 0x884A,
                 TEXTURE_STENCIL_SIZE = 0x88F1, TEXTURE_SHARED_SIZE = 0x8C3F, TEXTURE_RED_TYPE = 0x8C10,
                 TEXTURE_GREEN_TYPE = 0x8C11, TEXTURE_BLUE_TYPE = 0x8C12, TEXTURE_ALPHA_TYPE = 0x8C13,
                 TEXTURE_DEPTH_TYPE = 0x8C16, TEXTURE_COMPRESSED = 0x86A1, TEXTURE_FIXED_SAMPLE_LOCATIONS = 0x9107, TEXTURE_BUFFER_DATA_STORE_BINDING = 0x8C2D,
                 TEXTURE_BUFFER_OFFSET = 0x919D, TEXTURE_BUFFER_SIZE = 0x919E, MAX_TEXTURE_BUFFER_SIZE = 0x8C2B,
                 TEXTURE_BUFFER_OFFSET_ALIGNMENT = 0x919F, RENDERBUFFER_TARGET = 0x8D41;

// ---- images ------------------------------------------------------------------------------------------
constexpr GLenum READ_ONLY = 0x88B8, WRITE_ONLY = 0x88B9, READ_WRITE = 0x88BA, IMAGE_BINDING_NAME = 0x8F3A,
                 IMAGE_BINDING_LEVEL = 0x8F3B, IMAGE_BINDING_LAYERED = 0x8F3C, IMAGE_BINDING_LAYER = 0x8F3D,
                 IMAGE_BINDING_ACCESS = 0x8F3E, IMAGE_BINDING_FORMAT = 0x906E;

// ---- program interfaces -----------------------------------------------------------------------------
constexpr GLenum UNIFORM_IF = 0x92E1, UNIFORM_BLOCK_IF = 0x92E2, PROGRAM_INPUT = 0x92E3, PROGRAM_OUTPUT = 0x92E4,
                 BUFFER_VARIABLE = 0x92E5, SHADER_STORAGE_BLOCK_IF = 0x92E6, ATOMIC_COUNTER_BUFFER_IF = 0x92C0,
                 TRANSFORM_FEEDBACK_VARYING = 0x92F4, ACTIVE_RESOURCES = 0x92F5, MAX_NAME_LENGTH = 0x92F6,
                 MAX_NUM_ACTIVE_VARIABLES = 0x92F7, NAME_LENGTH = 0x92F9, TYPE = 0x92FA, ARRAY_SIZE = 0x92FB,
                 OFFSET = 0x92FC, BLOCK_INDEX = 0x92FD, ARRAY_STRIDE = 0x92FE, MATRIX_STRIDE = 0x92FF,
                 IS_ROW_MAJOR = 0x9300, ATOMIC_COUNTER_BUFFER_INDEX = 0x9301, BUFFER_BINDING = 0x9302,
                 BUFFER_DATA_SIZE = 0x9303, NUM_ACTIVE_VARIABLES = 0x9304, ACTIVE_VARIABLES = 0x9305,
                 REFERENCED_BY_VERTEX_SHADER = 0x9306, REFERENCED_BY_FRAGMENT_SHADER = 0x930A,
                 REFERENCED_BY_COMPUTE_SHADER = 0x930B, TOP_LEVEL_ARRAY_SIZE = 0x930C, TOP_LEVEL_ARRAY_STRIDE = 0x930D,
                 LOCATION = 0x930E, PROGRAM_SEPARABLE = 0x8258;

// ---- KHR_debug ---------------------------------------------------------------------------------------
constexpr GLenum DEBUG_SOURCE_API = 0x8246, DEBUG_SOURCE_WINDOW_SYSTEM = 0x8247, DEBUG_SOURCE_SHADER_COMPILER = 0x8248,
                 DEBUG_SOURCE_THIRD_PARTY = 0x8249, DEBUG_SOURCE_APPLICATION = 0x824A, DEBUG_SOURCE_OTHER = 0x824B,
                 DEBUG_TYPE_ERROR = 0x824C, DEBUG_TYPE_MARKER = 0x8268, DEBUG_TYPE_PUSH_GROUP = 0x8269,
                 DEBUG_TYPE_POP_GROUP = 0x826A, DEBUG_SEVERITY_HIGH = 0x9146, DEBUG_SEVERITY_NOTIFICATION = 0x826B,
                 DEBUG_CALLBACK_FUNCTION = 0x8244, DEBUG_CALLBACK_USER_PARAM = 0x8245,
                 DEBUG_NEXT_LOGGED_MESSAGE_LENGTH = 0x8243, DEBUG_LOGGED_MESSAGES = 0x9145,
                 DEBUG_GROUP_STACK_DEPTH = 0x826D, MAX_DEBUG_GROUP_STACK_DEPTH = 0x826C, MAX_DEBUG_LOGGED_MESSAGES = 0x9144,
                 MAX_LABEL_LENGTH = 0x82E8, STACK_OVERFLOW = 0x0503, STACK_UNDERFLOW = 0x0504, DONT_CARE_ENUM = 0x1100;
constexpr int kMaxDebugGroupDepth = 64, kMaxDebugLoggedMessages = 128, kMaxLabelLength = 256;
// Object identifiers of glObjectLabel.
constexpr GLenum BUFFER_ID = 0x82E0, SHADER_ID = 0x82E1, PROGRAM_ID = 0x82E2, VERTEX_ARRAY_ID = 0x8074, QUERY_ID = 0x82E3,
                 PROGRAM_PIPELINE_ID = 0x82E4, SAMPLER_ID = 0x82E6, TEXTURE_ID = 0x1702, RENDERBUFFER_ID = 0x8D41,
                 FRAMEBUFFER_ID = 0x8D40, TRANSFORM_FEEDBACK_ID = 0x8E22;
// EXT_debug_label identifiers (glLabelObjectEXT).
constexpr GLenum BUFFER_OBJECT_EXT = 0x9151, SHADER_OBJECT_EXT = 0x8B48, PROGRAM_OBJECT_EXT = 0x8B40,
                 VERTEX_ARRAY_OBJECT_EXT = 0x9154, QUERY_OBJECT_EXT = 0x9153, PROGRAM_PIPELINE_OBJECT_EXT = 0x8A4F;

// ---- sampler types (glGetActiveUniform) --------------------------------------------------------------
constexpr GLenum SAMPLER_BUFFER = 0x8DC2, INT_SAMPLER_BUFFER = 0x8DD0, UNSIGNED_INT_SAMPLER_BUFFER = 0x8DD8;

}  // namespace gl
}  // namespace vpgles
