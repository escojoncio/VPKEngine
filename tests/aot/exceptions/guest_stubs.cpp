// libstdc++ functions some builds of libsupc++ reference (the guest image has no libstdc++).
namespace std {
[[noreturn]] void __throw_out_of_range_fmt(const char*, ...) { __builtin_trap(); }
}
