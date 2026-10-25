#include "r3d/core/log.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>

namespace r3d {
namespace {

std::atomic<LogLevel> g_level{LogLevel::info};
std::mutex g_mutex;  // guards g_sink and serialises output
LogSink g_sink;
const auto g_start = std::chrono::steady_clock::now();

char level_letter(LogLevel level) {
  switch (level) {
    case LogLevel::trace: return 'T';
    case LogLevel::debug: return 'D';
    case LogLevel::info: return 'I';
    case LogLevel::warn: return 'W';
    case LogLevel::error: return 'E';
    case LogLevel::off: break;
  }
  return '?';
}

}  // namespace

void set_log_level(LogLevel level) { g_level.store(level, std::memory_order_relaxed); }

LogLevel log_level() { return g_level.load(std::memory_order_relaxed); }

void set_log_sink(LogSink sink) {
  const std::lock_guard lock(g_mutex);
  g_sink = std::move(sink);
}

void log_message(LogLevel level, std::string_view message) {
  const std::lock_guard lock(g_mutex);
  if (g_sink) {
    g_sink(level, message);
    return;
  }
  const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
  const std::string line = std::format("[{:9.3f}s] {} {}\n", elapsed, level_letter(level), message);
  std::fputs(line.c_str(), stderr);
}

std::string_view to_string(LogLevel level) {
  switch (level) {
    case LogLevel::trace: return "trace";
    case LogLevel::debug: return "debug";
    case LogLevel::info: return "info";
    case LogLevel::warn: return "warn";
    case LogLevel::error: return "error";
    case LogLevel::off: return "off";
  }
  return "unknown";
}

bool parse_log_level(std::string_view name, LogLevel& level) {
  for (const LogLevel candidate : {LogLevel::trace, LogLevel::debug, LogLevel::info, LogLevel::warn,
                                   LogLevel::error, LogLevel::off}) {
    if (name == to_string(candidate)) {
      level = candidate;
      return true;
    }
  }
  return false;
}

}  // namespace r3d
