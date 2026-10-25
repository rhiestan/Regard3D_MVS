#pragma once

#include <charconv>
#include <locale>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

namespace r3d::text {

// Locale-independent parsing for text formats (COLMAP models etc.).

[[nodiscard]] inline bool is_space(char c) noexcept { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

// Splits at runs of spaces / tabs; ignores a trailing '\r'.
[[nodiscard]] inline std::vector<std::string_view> split_ws(std::string_view line) {
  std::vector<std::string_view> tokens;
  std::size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() && is_space(line[i])) {
      ++i;
    }
    const std::size_t start = i;
    while (i < line.size() && !is_space(line[i])) {
      ++i;
    }
    if (i > start) {
      tokens.push_back(line.substr(start, i - start));
    }
  }
  return tokens;
}

[[nodiscard]] inline std::string_view trim(std::string_view s) noexcept {
  while (!s.empty() && is_space(s.front())) {
    s.remove_prefix(1);
  }
  while (!s.empty() && is_space(s.back())) {
    s.remove_suffix(1);
  }
  return s;
}

// The whole token must be a number; nullopt otherwise.
template <typename T>
[[nodiscard]] std::optional<T> parse_number(std::string_view token) {
  static_assert(std::is_arithmetic_v<T>);
  if (token.empty()) {
    return std::nullopt;
  }
#if !(defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L)
  if constexpr (std::is_floating_point_v<T>) {
    // Standard libraries without floating-point from_chars (older libc++).
    std::istringstream ss{std::string(token)};
    ss.imbue(std::locale::classic());
    T value{};
    ss >> value;
    if (ss.fail() || ss.peek() != std::char_traits<char>::eof()) {
      return std::nullopt;
    }
    return value;
  } else
#endif
  {
    T value{};
    const char* end = token.data() + token.size();
    const auto [ptr, ec] = std::from_chars(token.data(), end, value);
    if (ec != std::errc() || ptr != end) {
      return std::nullopt;
    }
    return value;
  }
}

}  // namespace r3d::text
