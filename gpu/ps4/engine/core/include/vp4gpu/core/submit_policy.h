// When recorded work becomes a GPU frame.
//
// Every submit is a frame start for the frame compiler (a tile load/store per render target, a
// command buffer chained to the previous by an event) and a trip through the encoder. A title that
// ends a command buffer ~15 times per picture must not pay that 15 times: work accumulates into one
// Frame and is handed to the GPU only when something needs it to run:
//   - the picture is presented (flip),
//   - the CPU is about to touch memory the recorded work reads or writes (DMA, GDS, WRITE_DATA),
//   - a WAIT_REG_MEM waits for a label the recorded work (or a pending label write) produces,
//   - a label is waiting on the recorded work for too long (the title may poll it),
//   - the frame grew past a size bound or has been open for too long.
// Pure decision logic: the host feeds it the state and a clock, so it is testable without a GPU.
#pragma once

#include <algorithm>
#include <cstdint>

namespace vp4gpu {

enum class SubmitReason : uint8_t { Flip, Dma, Gds, Finish, WaitLabel, WriteData, LabelGrace, Size, Age, Idle, Count };
inline const char* toString(SubmitReason r) {
  static const char* n[] = {"flip", "dma", "gds", "finish", "wait_label", "write_data", "label_grace", "size", "age", "idle"};
  return n[static_cast<int>(r)];
}

struct SubmitPolicy {
  uint32_t maxCommands = 20000;   // recorded commands (draws + dispatches + copies) before a forced submit
  double maxOpenMs = 40;          // a frame never stays open longer than this (>= one 25 fps picture)
  double labelGraceMs = 6;        // a label write waiting on unsubmitted work
  double labelGraceIrqMs = 1;     // the same when it raises an interrupt (a thread sleeps on it)
};

struct SubmitState {
  uint64_t commands = 0;         // recorded and not submitted
  double openMs = 0;             // since the first of them
  bool labelWaiting = false;     // a label write depends on the recorded work
  double labelMs = 0;            // age of the oldest such label
  bool labelIrq = false;         // ... and one of them raises an interrupt
};

// The reason to submit now, or none (Count). `commands == 0` never submits.
inline SubmitReason decide(const SubmitPolicy& p, const SubmitState& s) {
  if (s.commands == 0) return SubmitReason::Count;
  if (s.commands >= p.maxCommands) return SubmitReason::Size;
  if (s.labelWaiting && s.labelMs >= (s.labelIrq ? p.labelGraceIrqMs : p.labelGraceMs)) return SubmitReason::LabelGrace;
  if (s.openMs >= p.maxOpenMs) return SubmitReason::Age;
  return SubmitReason::Count;
}

// Milliseconds until decide() would say submit for a state that only ages (the watchdog's sleep);
// a negative value: nothing to wait for.
inline double nextCheckMs(const SubmitPolicy& p, const SubmitState& s) {
  if (s.commands == 0) return -1;
  double t = p.maxOpenMs - s.openMs;
  if (s.labelWaiting) t = std::min(t, (s.labelIrq ? p.labelGraceIrqMs : p.labelGraceMs) - s.labelMs);
  return std::max(t, 0.0);
}

}  // namespace vp4gpu
