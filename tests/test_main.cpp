#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#if defined(_MSC_VER)
#include <crtdbg.h>
#include <cstdlib>
#include <initializer_list>
#endif

int main(int argc, char** argv) {
#if defined(_MSC_VER)
  // Report CRT assertions, abort() and invalid-parameter errors on stderr instead of in a modal
  // dialog, which would block unattended test runs (CI, ctest) indefinitely.
  for ([[maybe_unused]] const int type : {_CRT_WARN, _CRT_ERROR, _CRT_ASSERT}) {
    _CrtSetReportMode(type, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(type, _CRTDBG_FILE_STDERR);
  }
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif

  doctest::Context context(argc, argv);
  return context.run();
}
