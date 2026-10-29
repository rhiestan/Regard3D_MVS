#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace r3d {

// Fixed-size pool of worker threads with a blocking parallel_for.
//
// Determinism: work is split into chunks whose boundaries depend only on
// (begin, end, grain), never on the thread count or timing. A body that
// writes only to its own chunk therefore gives identical results with any
// number of threads.
//
// Cancellation: before every chunk, parallel_for polls the cancellation
// token of the calling thread (see cancel.hpp) and stops with Cancelled once
// it is requested; bodies run with the caller's token.
//
// Nesting: parallel_for called from inside a parallel_for body runs serially
// on the calling thread. This avoids deadlock and oversubscription; outer
// loops should carry the parallelism.
class ThreadPool {
public:
  // `num_threads` counts all threads that execute work, including the thread
  // calling parallel_for; 0 selects std::thread::hardware_concurrency().
  explicit ThreadPool(unsigned num_threads = 0);
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ~ThreadPool();

  [[nodiscard]] unsigned num_threads() const noexcept { return static_cast<unsigned>(workers_.size()) + 1; }

  // Calls body(chunk_begin, chunk_end) for consecutive chunks of [begin, end)
  // with `grain` elements each (the last may be shorter) and waits for all of
  // them. The first exception thrown by a body is rethrown here; remaining
  // chunks are skipped once a body has failed.
  template <typename Body>
  void parallel_for(std::size_t begin, std::size_t end, std::size_t grain, Body&& body) {
    const std::function<void(std::size_t, std::size_t)> fn = std::forward<Body>(body);
    run(begin, end, grain, fn);
  }

  // True when called from inside a parallel_for body of any pool.
  [[nodiscard]] static bool in_parallel_region() noexcept;

private:
  void run(std::size_t begin, std::size_t end, std::size_t grain,
           const std::function<void(std::size_t, std::size_t)>& body);
  void enqueue(std::function<void()> task);
  void worker_loop();

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> tasks_;
  bool stop_ = false;
};

// Process-wide pool used by the library. It is created on first use with
// hardware_concurrency() threads unless set_default_thread_count() is called
// before that.
ThreadPool& default_thread_pool();

// Recreates the default pool with `num_threads` threads (0 = hardware).
// Must not be called while the default pool is executing work.
void set_default_thread_count(unsigned num_threads);

template <typename Body>
void parallel_for(std::size_t begin, std::size_t end, std::size_t grain, Body&& body) {
  default_thread_pool().parallel_for(begin, end, grain, std::forward<Body>(body));
}

}  // namespace r3d
