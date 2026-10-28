#include "r3d/core/memory.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// windows.h first
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace r3d {

std::size_t peak_memory_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) == 0) {
    return 0;
  }
  return counters.PeakWorkingSetSize;
#else
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0;
  }
#if defined(__APPLE__)
  return static_cast<std::size_t>(usage.ru_maxrss);  // bytes
#else
  return static_cast<std::size_t>(usage.ru_maxrss) * 1024;  // kilobytes
#endif
#endif
}

}  // namespace r3d
