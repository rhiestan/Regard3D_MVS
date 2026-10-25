#include <atomic>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include <doctest/doctest.h>

#include "r3d/core/thread_pool.hpp"

using r3d::ThreadPool;

TEST_CASE("parallel_for visits every index exactly once") {
  for (const unsigned threads : {1u, 2u, 4u, 9u}) {
    for (const std::size_t grain : {std::size_t{1}, std::size_t{7}, std::size_t{1000}}) {
      CAPTURE(threads);
      CAPTURE(grain);
      ThreadPool pool(threads);
      CHECK(pool.num_threads() == threads);

      constexpr std::size_t kBegin = 3;
      constexpr std::size_t kEnd = 2503;
      std::vector<std::atomic<int>> visits(kEnd);
      pool.parallel_for(kBegin, kEnd, grain, [&](std::size_t b, std::size_t e) {
        CHECK(b < e);
        CHECK(e - b <= grain);
        CHECK((b - kBegin) % grain == 0);  // chunk boundaries independent of threads
        for (std::size_t i = b; i < e; ++i) {
          visits[i].fetch_add(1);
        }
      });
      for (std::size_t i = 0; i < kEnd; ++i) {
        CHECK(visits[i].load() == (i >= kBegin ? 1 : 0));
      }
    }
  }
}

TEST_CASE("parallel_for handles empty ranges") {
  ThreadPool pool(4);
  bool called = false;
  pool.parallel_for(10, 10, 1, [&](std::size_t, std::size_t) { called = true; });
  pool.parallel_for(10, 5, 1, [&](std::size_t, std::size_t) { called = true; });
  CHECK_FALSE(called);
}

TEST_CASE("parallel_for propagates exceptions") {
  ThreadPool pool(4);
  std::atomic<int> executed{0};
  CHECK_THROWS_AS(pool.parallel_for(0, 1000, 1,
                                    [&](std::size_t b, std::size_t) {
                                      executed.fetch_add(1);
                                      if (b == 17) {
                                        throw std::runtime_error("boom");
                                      }
                                    }),
                  std::runtime_error);
  // The pool must stay usable after a failure.
  std::atomic<std::size_t> sum{0};
  pool.parallel_for(0, 100, 10, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) sum += i;
  });
  CHECK(sum.load() == 4950);
}

TEST_CASE("nested parallel_for runs serially without deadlock") {
  ThreadPool pool(4);
  std::atomic<int> total{0};
  pool.parallel_for(0, 8, 1, [&](std::size_t, std::size_t) {
    CHECK(ThreadPool::in_parallel_region());
    pool.parallel_for(0, 100, 3, [&](std::size_t b, std::size_t e) { total += static_cast<int>(e - b); });
  });
  CHECK(total.load() == 800);
  CHECK_FALSE(ThreadPool::in_parallel_region());
}

TEST_CASE("concurrent callers share one pool") {
  ThreadPool pool(4);
  std::vector<std::thread> callers;
  std::vector<std::size_t> sums(6, 0);
  for (std::size_t t = 0; t < sums.size(); ++t) {
    callers.emplace_back([&, t] {
      std::vector<std::size_t> partial(50, 0);
      pool.parallel_for(0, 50, 1, [&](std::size_t b, std::size_t) { partial[b] = b * (t + 1); });
      sums[t] = std::accumulate(partial.begin(), partial.end(), std::size_t{0});
    });
  }
  for (std::thread& c : callers) c.join();
  for (std::size_t t = 0; t < sums.size(); ++t) {
    CHECK(sums[t] == 1225 * (t + 1));
  }
}

TEST_CASE("default pool can be resized") {
  r3d::set_default_thread_count(3);
  CHECK(r3d::default_thread_pool().num_threads() == 3);
  std::atomic<int> count{0};
  r3d::parallel_for(0, 10, 1, [&](std::size_t, std::size_t) { ++count; });
  CHECK(count.load() == 10);
  r3d::set_default_thread_count(0);
  CHECK(r3d::default_thread_pool().num_threads() >= 1);
}
