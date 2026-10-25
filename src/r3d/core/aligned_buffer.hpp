#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

namespace r3d {

// Alignment used for all bulk buffers: one cache line, and enough for any
// SIMD vector up to AVX-512.
inline constexpr std::size_t kBufferAlignment = 64;

// Owning, cache-line aligned array of trivially copyable elements.
// Unlike std::vector, resizing does not initialise elements (hot buffers are
// usually fully overwritten); use fill() when initialisation is required.
template <typename T>
class AlignedBuffer {
  static_assert(std::is_trivially_copyable_v<T>, "AlignedBuffer requires trivially copyable T");

public:
  AlignedBuffer() = default;
  explicit AlignedBuffer(std::size_t size) { reset_uninitialized(size); }

  AlignedBuffer(const AlignedBuffer& other) {
    reset_uninitialized(other.size_);
    if (size_ > 0) {
      std::memcpy(data_, other.data_, size_ * sizeof(T));
    }
  }

  AlignedBuffer(AlignedBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

  AlignedBuffer& operator=(const AlignedBuffer& other) {
    if (this != &other) {
      AlignedBuffer copy(other);
      swap(copy);
    }
    return *this;
  }

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  ~AlignedBuffer() { release(); }

  // Discards the current contents and allocates `size` uninitialised elements.
  void reset_uninitialized(std::size_t size) {
    if (size == size_) {
      return;
    }
    release();
    if (size > 0) {
      data_ = static_cast<T*>(::operator new(size * sizeof(T), std::align_val_t{kBufferAlignment}));
      size_ = size;
    }
  }

  void fill(const T& value) { std::fill_n(data_, size_, value); }

  void swap(AlignedBuffer& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  T& operator[](std::size_t i) noexcept { return data_[i]; }
  const T& operator[](std::size_t i) const noexcept { return data_[i]; }

  T* begin() noexcept { return data_; }
  T* end() noexcept { return data_ + size_; }
  const T* begin() const noexcept { return data_; }
  const T* end() const noexcept { return data_ + size_; }

private:
  void release() noexcept {
    if (data_ != nullptr) {
      ::operator delete(data_, std::align_val_t{kBufferAlignment});
      data_ = nullptr;
      size_ = 0;
    }
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace r3d
