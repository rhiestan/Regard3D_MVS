#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "r3d/core/simd.cpp"
#include <hwy/foreach_target.h>  // IWYU pragma: keep

#include <hwy/highway.h>

#include "r3d/core/simd.hpp"

HWY_BEFORE_NAMESPACE();
namespace r3d {
namespace HWY_NAMESPACE {

// Returns the target this copy was compiled for; dispatching to it reveals
// which target the runtime dispatcher selects.
std::int64_t CompiledTarget() { return HWY_TARGET; }

}  // namespace HWY_NAMESPACE
}  // namespace r3d
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace r3d {

HWY_EXPORT(CompiledTarget);

std::string simd_active_target_name() { return simd_target_name(HWY_DYNAMIC_DISPATCH(CompiledTarget)()); }

std::vector<std::int64_t> simd_available_targets() { return hwy::SupportedAndGeneratedTargets(); }

std::string simd_target_name(std::int64_t target) { return hwy::TargetName(target); }

ScopedSimdTarget::ScopedSimdTarget(std::int64_t target) { hwy::SetSupportedTargetsForTest(target); }

ScopedSimdTarget::~ScopedSimdTarget() { hwy::SetSupportedTargetsForTest(0); }

}  // namespace r3d

#endif  // HWY_ONCE
