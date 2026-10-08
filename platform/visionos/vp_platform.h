/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * The C side of the shared visionOS layer: what a game core receives from the Swift app.
 * Mirrors VPControllerState in Sources/VPPlatform.swift; the game app converts between them.
 */
#ifndef VP_PLATFORM_H
#define VP_PLATFORM_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vp_controller {
    bool active;
    float move_x, move_y;   /* left stick, -1..1 */
    float turn_x, turn_y;   /* right stick, -1..1 */
    bool cross, circle, square, triangle;
    bool options, touchpad;
    bool l1, r1;
    float l2, r2;           /* triggers, 0..1 */
    bool l3, r3;
    bool dpad_up, dpad_down, dpad_left, dpad_right;
    bool hand_valid[2];
    float hand_pos[2][3];
    float hand_rot[2][4];   /* quaternion x, y, z, w */
    int prompt_style;       /* 0 Xbox, 1 PlayStation, 2 Nintendo */
} vp_controller;

#ifdef __cplusplus
}
#endif
#endif
