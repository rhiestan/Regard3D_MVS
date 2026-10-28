#include "r3d/mvs/depth_map_store.hpp"

#include <system_error>
#include <utility>

#include "r3d/core/error.hpp"

namespace r3d {

std::size_t depth_map_bytes(const DepthMap& map) {
  // depth, confidence (float), pair mask (uint16), support (uint8), colour (3 x uint8)
  return static_cast<std::size_t>(map.width()) * static_cast<std::size_t>(map.height()) * 14 + sizeof(DepthMap);
}

DepthMapStore::DepthMapStore(std::size_t count, PathFn path, std::size_t budget, bool persistent)
    : slots_(count), path_(std::move(path)), budget_(budget), persistent_(persistent) {}

DepthMapStore::~DepthMapStore() {
  if (persistent_) {
    return;
  }
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    if (slots_[i].on_disk) {
      std::error_code ec;
      std::filesystem::remove(path_(i), ec);
    }
  }
}

void DepthMapStore::touch(std::size_t i) {
  Slot& s = slots_[i];
  if (s.in_lru) {
    lru_.erase(s.lru);
  }
  lru_.push_front(i);
  s.lru = lru_.begin();
  s.in_lru = true;
}

void DepthMapStore::evict_locked() {
  // Keeps at least the most recently used map.
  while (resident_ > budget_ && lru_.size() > 1) {
    const std::size_t i = lru_.back();
    lru_.pop_back();
    Slot& s = slots_[i];
    s.in_lru = false;
    if (!s.on_disk) {
      write_depth_map(path_(i), *s.map);
      s.on_disk = true;
    }
    s.map.reset();
    resident_ -= s.bytes;
  }
}

void DepthMapStore::put(std::size_t i, DepthMap map) {
  if (i >= slots_.size()) {
    throw_error("depth map store: view {} of {}", i, slots_.size());
  }
  auto shared = std::make_shared<const DepthMap>(std::move(map));
  const std::lock_guard lock(mutex_);
  Slot& s = slots_[i];
  if (s.map) {
    resident_ -= s.bytes;
  }
  s.on_disk = false;
  if (persistent_ && shared->width() > 0) {
    write_depth_map(path_(i), *shared);
    s.on_disk = true;
  }
  s.map = std::move(shared);
  s.stored = true;
  s.bytes = depth_map_bytes(*s.map);
  resident_ += s.bytes;
  touch(i);
  evict_locked();
}

void DepthMapStore::adopt_file(std::size_t i) {
  const std::lock_guard lock(mutex_);
  Slot& s = slots_[i];
  if (s.map) {
    resident_ -= s.bytes;
    s.map.reset();
  }
  if (s.in_lru) {
    lru_.erase(s.lru);
    s.in_lru = false;
  }
  s.stored = true;
  s.on_disk = true;
}

std::shared_ptr<const DepthMap> DepthMapStore::get(std::size_t i) {
  if (i >= slots_.size()) {
    throw_error("depth map store: view {} of {}", i, slots_.size());
  }
  const std::lock_guard lock(mutex_);
  Slot& s = slots_[i];
  if (!s.stored) {
    return std::make_shared<const DepthMap>();
  }
  if (!s.map) {
    s.map = std::make_shared<const DepthMap>(read_depth_map(path_(i)));
    s.bytes = depth_map_bytes(*s.map);
    resident_ += s.bytes;
  }
  std::shared_ptr<const DepthMap> result = s.map;
  touch(i);
  evict_locked();
  return result;
}

bool DepthMapStore::contains(std::size_t i) const {
  const std::lock_guard lock(mutex_);
  return i < slots_.size() && slots_[i].stored;
}

std::size_t DepthMapStore::resident_bytes() const {
  const std::lock_guard lock(mutex_);
  return resident_;
}

}  // namespace r3d
