// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// What vpconvert keeps between its calls in one run of the app, so that it does not read again
// what it already read. No LLVM here: tests/rt/vpconvert_cache.cpp builds it alone.
#ifndef VPCONVERT_CACHE_H
#define VPCONVERT_CACHE_H

#include <cstdio>
#include <map>
#include <mutex>
#include <string>

#include <sys/stat.h>

namespace vpcache {

// Size and modification time of a file (to the nanosecond where the system gives it).
struct Stamp {
    long long size = 0, sec = 0, nsec = 0;
    bool operator==(const Stamp& o) const { return size == o.size && sec == o.sec && nsec == o.nsec; }
};

inline bool stamp_of(const std::string& path, Stamp& out) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) return false;
    out.size = (long long)st.st_size;
#ifdef __APPLE__
    out.sec = (long long)st.st_mtimespec.tv_sec;
    out.nsec = (long long)st.st_mtimespec.tv_nsec;
#else
    out.sec = (long long)st.st_mtim.tv_sec;
    out.nsec = (long long)st.st_mtim.tv_nsec;
#endif
    return true;
}

// A value worked out from a file's contents (the hash of a header of the SDK), kept while the
// file's size and modification time are the same. Any thread.
class FileValues {
public:
    template <class F> std::string get(const std::string& path, F compute) {
        Stamp now;
        if (!stamp_of(path, now)) return compute();
        {
            std::scoped_lock lock{mutex_};
            const auto it = values_.find(path);
            if (it != values_.end() && it->second.stamp == now) return it->second.value;
        }
        std::string value = compute();
        std::scoped_lock lock{mutex_};
        values_[path] = {now, value};
        return value;
    }

private:
    struct Entry { Stamp stamp; std::string value; };
    std::mutex mutex_;
    std::map<std::string, Entry> values_;
};

// The embedder's patch of a module's image (read from the module and run through the embedder's
// callback), by module file, name and callback. `recompute` asks for it to be worked out again (a
// conversion, or the check whether the pack is current: the embedder's settings may have changed
// since) and kept; without it, the value kept is used while the module file is the same (a
// translation on demand while the game runs, one per code the game reaches: the game's patches do
// not change meanwhile). A `compute` that throws keeps nothing. Any thread.
class Patches {
public:
    template <class F>
    std::string get(const std::string& file, const std::string& name, const void* callback, const void* user, bool recompute, F compute) {
        char ids[48];
        snprintf(ids, sizeof ids, "%p %p", callback, user);
        const std::string key = file + '\n' + name + '\n' + ids;
        Stamp now;
        const bool have_stamp = stamp_of(file, now);
        if (!recompute && have_stamp) {
            std::scoped_lock lock{mutex_};
            const auto it = values_.find(key);
            if (it != values_.end() && it->second.stamp == now) return it->second.value;
        }
        std::string value = compute();
        if (have_stamp) {
            std::scoped_lock lock{mutex_};
            values_[key] = {now, value};
        }
        return value;
    }

private:
    struct Entry { Stamp stamp; std::string value; };
    std::mutex mutex_;
    std::map<std::string, Entry> values_;
};

} // namespace vpcache

#endif
