#pragma once

#include <format>
#include <stdexcept>
#include <string>
#include <utility>

namespace r3d {

// Base exception for all errors reported by the library. Exceptions are used
// at API boundaries (I/O, invalid input), never inside hot loops.
class Error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class IoError : public Error {
public:
  using Error::Error;
};

template <typename... Args>
[[noreturn]] void throw_error(std::format_string<Args...> fmt, Args&&... args) {
  throw Error(std::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
[[noreturn]] void throw_io_error(std::format_string<Args...> fmt, Args&&... args) {
  throw IoError(std::format(fmt, std::forward<Args>(args)...));
}

}  // namespace r3d
