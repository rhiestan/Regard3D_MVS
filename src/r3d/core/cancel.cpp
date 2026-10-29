#include "r3d/core/cancel.hpp"

namespace r3d {
namespace {

thread_local const CancellationToken* t_token = nullptr;

}  // namespace

const CancellationToken* current_cancellation() noexcept { return t_token; }

void throw_if_cancelled() {
  if (t_token != nullptr) {
    t_token->throw_if_requested();
  }
}

CancellationScope::CancellationScope(const CancellationToken* token) noexcept : previous_(t_token) { t_token = token; }

CancellationScope::~CancellationScope() { t_token = previous_; }

}  // namespace r3d
