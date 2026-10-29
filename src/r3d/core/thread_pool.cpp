#include "r3d/core/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>

#include "r3d/core/cancel.hpp"

namespace r3d {
namespace {

thread_local int t_parallel_depth = 0;

struct DepthGuard {
  DepthGuard() { ++t_parallel_depth; }
  DepthGuard(const DepthGuard&) = delete;
  DepthGuard& operator=(const DepthGuard&) = delete;
  ~DepthGuard() { --t_parallel_depth; }
};

// Shared state of one parallel_for call. Owned by shared_ptr because helper
// tasks may be dequeued after the call has already returned; such late tasks
// find no chunk left and never touch `body`.
struct Job {
  std::size_t begin = 0;
  std::size_t end = 0;
  std::size_t grain = 1;
  std::size_t num_chunks = 0;
  const std::function<void(std::size_t, std::size_t)>* body = nullptr;
  const CancellationToken* cancel = nullptr;  // of the calling thread

  std::atomic<std::size_t> next_chunk{0};
  std::atomic<std::size_t> finished_chunks{0};
  std::atomic<bool> failed{false};

  std::mutex mutex;
  std::condition_variable done_cv;
  std::exception_ptr error;

  void execute() {
    const DepthGuard depth;
    const CancellationScope scope(cancel);
    for (;;) {
      const std::size_t chunk = next_chunk.fetch_add(1, std::memory_order_relaxed);
      if (chunk >= num_chunks) {
        return;
      }
      if (!failed.load(std::memory_order_relaxed)) {
        const std::size_t chunk_begin = begin + chunk * grain;
        const std::size_t chunk_end = std::min(end, chunk_begin + grain);
        try {
          if (cancel != nullptr) {
            cancel->throw_if_requested();
          }
          (*body)(chunk_begin, chunk_end);
        } catch (...) {
          const std::lock_guard lock(mutex);
          if (!error) {
            error = std::current_exception();
          }
          failed.store(true, std::memory_order_relaxed);
        }
      }
      if (finished_chunks.fetch_add(1, std::memory_order_acq_rel) + 1 == num_chunks) {
        const std::lock_guard lock(mutex);
        done_cv.notify_all();
      }
    }
  }
};

std::mutex g_default_pool_mutex;
std::unique_ptr<ThreadPool> g_default_pool;

}  // namespace

ThreadPool::ThreadPool(unsigned num_threads) {
  if (num_threads == 0) {
    num_threads = std::max(1u, std::thread::hardware_concurrency());
  }
  workers_.reserve(num_threads - 1);
  for (unsigned i = 1; i < num_threads; ++i) {
    workers_.emplace_back([this] { worker_loop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    const std::lock_guard lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  for (std::thread& worker : workers_) {
    worker.join();
  }
}

bool ThreadPool::in_parallel_region() noexcept { return t_parallel_depth > 0; }

void ThreadPool::run(std::size_t begin, std::size_t end, std::size_t grain,
                     const std::function<void(std::size_t, std::size_t)>& body) {
  if (end <= begin) {
    return;
  }
  grain = std::max<std::size_t>(grain, 1);
  const std::size_t num_chunks = (end - begin + grain - 1) / grain;

  // Serial path: single chunk, no workers, or nested call. Chunk boundaries
  // are the same as in the parallel path.
  if (num_chunks == 1 || workers_.empty() || in_parallel_region()) {
    const DepthGuard depth;
    for (std::size_t chunk_begin = begin; chunk_begin < end; chunk_begin += grain) {
      throw_if_cancelled();
      body(chunk_begin, std::min(end, chunk_begin + grain));
    }
    return;
  }

  auto job = std::make_shared<Job>();
  job->begin = begin;
  job->end = end;
  job->grain = grain;
  job->num_chunks = num_chunks;
  job->body = &body;
  job->cancel = current_cancellation();

  const std::size_t helpers = std::min<std::size_t>(workers_.size(), num_chunks - 1);
  for (std::size_t i = 0; i < helpers; ++i) {
    enqueue([job] { job->execute(); });
  }
  job->execute();

  {
    std::unique_lock lock(job->mutex);
    job->done_cv.wait(lock, [&] { return job->finished_chunks.load(std::memory_order_acquire) == num_chunks; });
  }
  if (job->error) {
    std::rethrow_exception(job->error);
  }
}

void ThreadPool::enqueue(std::function<void()> task) {
  {
    const std::lock_guard lock(mutex_);
    tasks_.push_back(std::move(task));
  }
  cv_.notify_one();
}

void ThreadPool::worker_loop() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
      if (tasks_.empty()) {
        return;  // stop_ requested and nothing left to do
      }
      task = std::move(tasks_.front());
      tasks_.pop_front();
    }
    task();
  }
}

ThreadPool& default_thread_pool() {
  const std::lock_guard lock(g_default_pool_mutex);
  if (!g_default_pool) {
    g_default_pool = std::make_unique<ThreadPool>();
  }
  return *g_default_pool;
}

void set_default_thread_count(unsigned num_threads) {
  const std::lock_guard lock(g_default_pool_mutex);
  g_default_pool = std::make_unique<ThreadPool>(num_threads);
}

}  // namespace r3d
