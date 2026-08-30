#include "ddx.hpp"

#include <gtest/gtest.h>

#include <format>

// One number, CMake's: the macros, the constants and the string agree.
static_assert(ddx::version_number == ddx::version_major * 100000 +
                                         ddx::version_minor * 100 +
                                         ddx::version_patch);
static_assert(DDX_VERSION == ddx::version_number);
static_assert(!ddx::version.empty());

TEST(Version, TheStringSpellsTheNumbers) {
  EXPECT_EQ(ddx::version, std::format("{}.{}.{}", ddx::version_major,
                                      ddx::version_minor, ddx::version_patch));
  EXPECT_EQ(ddx::version, DDX_VERSION_STRING);
}
