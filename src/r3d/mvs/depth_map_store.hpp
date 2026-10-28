#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

#include "r3d/mvs/depth_map.hpp"

namespace r3d {

// The depth maps of all views of one pipeline stage within a memory budget.
//
// Maps stay in memory while they fit into `budget` bytes; beyond it, the
// least recently used maps leave memory and are read from their file again
// when needed. With `persistent`, every map is written to its file when it
// is stored (the work-directory cache); otherwise a map is written only when
// it leaves memory, and the files are removed with the store.
class DepthMapStore {
 public:
  using PathFn = std::function<std::filesystem::path(std::size_t)>;

  DepthMapStore(std::size_t count, PathFn path, std::size_t budget, bool persistent);
  ~DepthMapStore();
  DepthMapStore(const DepthMapStore&) = delete;
  DepthMapStore& operator=(const DepthMapStore&) = delete;

  [[nodiscard]] std::size_t size() const noexcept { return slots_.size(); }
  // Stores the map of view i (an empty map marks a view without depths).
  void put(std::size_t i, DepthMap map);
  // The map of view i; empty if none was stored.
  [[nodiscard]] std::shared_ptr<const DepthMap> get(std::size_t i);
  // Marks view i as stored in its (persistent) file without loading it.
  void adopt_file(std::size_t i);
  [[nodiscard]] bool contains(std::size_t i) const;
  // Bytes of the maps in memory.
  [[nodiscard]] std::size_t resident_bytes() const;

 private:
  struct Slot {
    std::shared_ptr<const DepthMap> map;  // null: not in memory
    bool stored = false;
    bool on_disk = false;
    std::size_t bytes = 0;
    std::list<std::size_t>::iterator lru;
    bool in_lru = false;
  };
  void touch(std::size_t i);
  void evict_locked();

  std::vector<Slot> slots_;
  PathFn path_;
  std::size_t budget_;
  bool persistent_;
  std::size_t resident_ = 0;
  std::list<std::size_t> lru_;  // front: most recently used
  mutable std::mutex mutex_;
};

// Memory of a depth map's images in bytes.
[[nodiscard]] std::size_t depth_map_bytes(const DepthMap& map);

}  // namespace r3d
