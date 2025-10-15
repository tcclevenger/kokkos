// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <TestDefaultDeviceType_Category.hpp>

namespace Test {

void simple_range_loop() {
  Kokkos::parallel_for(
      Kokkos::RangePolicy(0, 1),
      KOKKOS_LAMBDA(const int i) { printf("%d\n", i); });
}

TEST(defaultdevicetype, development_test) { simple_range_loop(); }

}  // namespace Test
