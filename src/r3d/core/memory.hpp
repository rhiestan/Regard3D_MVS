#pragma once

#include <cstddef>

namespace r3d {

// Peak resident memory (working set) of this process in bytes; 0 if unknown.
[[nodiscard]] std::size_t peak_memory_bytes();

}  // namespace r3d
