// Stack budget of the frontends (SPIR-V lowering, GCN lifter). Their recursion over the structured control flow is bounded by
// a level count, but how deep a level reaches depends on the frame sizes of the compiler and the architecture (and the sanitizers),
// so the count alone cannot promise that a worker thread with a small stack survives. A StackGuard measures the stack of the
// current thread when the frontend starts and fixes a floor: stackLow() is true once the stack pointer is below it.
// Where the thread's stack size cannot be read (other platforms), stackLow() is always false and the level counts stay the only bound.
#pragma once

namespace vpir {

class StackGuard {
public:
    // Reserve kept free below the floor for the code that runs under a check (instruction handlers, containers, the
    // sanitizer): at least kMinReserve bytes, and a quarter of the stack that was available when the guard was made.
    static constexpr unsigned long kMinReserve = 32 * 1024;
    StackGuard();
    ~StackGuard();
    StackGuard(const StackGuard&) = delete;
    StackGuard& operator=(const StackGuard&) = delete;

private:
    unsigned long previous_;
};

// Is the stack of this thread, from the innermost StackGuard, nearly used up? (Cheap: one comparison.)
bool stackLow();

}  // namespace vpir
