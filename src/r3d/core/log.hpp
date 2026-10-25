#pragma once

#include <format>
#include <functional>
#include <string_view>
#include <utility>

namespace r3d {

enum class LogLevel { trace = 0, debug, info, warn, error, off };

// Messages below the threshold are discarded before formatting.
void set_log_level(LogLevel level);
[[nodiscard]] LogLevel log_level();
[[nodiscard]] inline bool log_enabled(LogLevel level) { return level >= log_level() && level != LogLevel::off; }

// Replaces the default stderr sink, e.g. to forward messages to a GUI.
// Passing an empty function restores the default sink. Sinks may be called
// from any thread, but never concurrently.
using LogSink = std::function<void(LogLevel, std::string_view)>;
void set_log_sink(LogSink sink);

void log_message(LogLevel level, std::string_view message);

[[nodiscard]] std::string_view to_string(LogLevel level);
// Accepts "trace", "debug", "info", "warn", "error", "off". Returns false on unknown names.
bool parse_log_level(std::string_view name, LogLevel& level);

template <typename... Args>
void log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
  if (log_enabled(level)) {
    log_message(level, std::format(fmt, std::forward<Args>(args)...));
  }
}

template <typename... Args>
void log_debug(std::format_string<Args...> fmt, Args&&... args) {
  log(LogLevel::debug, fmt, std::forward<Args>(args)...);
}
template <typename... Args>
void log_info(std::format_string<Args...> fmt, Args&&... args) {
  log(LogLevel::info, fmt, std::forward<Args>(args)...);
}
template <typename... Args>
void log_warn(std::format_string<Args...> fmt, Args&&... args) {
  log(LogLevel::warn, fmt, std::forward<Args>(args)...);
}
template <typename... Args>
void log_error(std::format_string<Args...> fmt, Args&&... args) {
  log(LogLevel::error, fmt, std::forward<Args>(args)...);
}

}  // namespace r3d
