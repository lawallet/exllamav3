#pragma once

// Explicitly enabled, bounded CPU-helper diagnostics. No shared protocol state,
// CUDA work, or arithmetic is changed. Times are inclusive steady-clock ns.
#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tp_profile
{
enum Field : size_t
{
    queue_begin, ready, done, bytes, mask, wire, math_ns, published,
    sleeps, sleep_ns, chunks, tier, arrivals, count = arrivals + 16
};
using Row = std::array<uint64_t, count>;
inline thread_local bool enabled = false;
inline thread_local size_t limit = 0;
inline thread_local uint64_t dropped = 0;
inline thread_local std::vector<Row> rows;
inline thread_local Row* current = nullptr;

inline uint64_t clock_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void begin(size_t capacity)
{
    if (current || enabled) throw std::runtime_error("TP profiling already active");
    if (capacity == 0 || capacity > 1048576) throw std::runtime_error("Invalid TP profile capacity");
    rows.clear();
    rows.reserve(capacity); // Only outside the measured dispatch loop.
    limit = capacity;
    dropped = 0;
    enabled = true;
}
inline std::vector<Row> end()
{
    if (current) throw std::runtime_error("Cannot stop TP profiling within a job");
    enabled = false;
    return rows;
}
struct Job
{
    Row* row = nullptr;
    Job()
    {
        if (!enabled) return;
        if (rows.size() >= limit) { ++dropped; return; }
        rows.emplace_back(Row{});
        row = current = &rows.back();
        (*row)[tier] = UINT64_MAX; // Distinguish an unmeasured dispatch/end marker.
        (*row)[queue_begin] = clock_ns();
    }
    ~Job()
    {
        if (row) { (*row)[done] = clock_ns(); current = nullptr; }
    }
};
struct Duration
{
    Row* row;
    Field field;
    uint64_t start;
    Duration(Row* r, Field f) : row(r), field(f), start(r ? clock_ns() : 0) {}
    ~Duration() { if (row) (*row)[field] += clock_ns() - start; }
};
}
