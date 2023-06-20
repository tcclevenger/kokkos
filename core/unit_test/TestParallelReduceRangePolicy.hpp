//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cstdio>

// This test checks parallel_reduce() calls which use RangePolicy.

namespace {

template <typename ValueType>
struct TestParallelReduceRangePolicy {
  // This typedef is needed for parallel_reduce() where a
  // work count is given (instead of a RangePolicy) so
  // that the execution space can be deduced internally.
  using execution_space = TEST_EXECSPACE; // maybe not?

  using ViewType = Kokkos::View<ValueType*, execution_space>;

  ViewType results;

  // Operator defining work done in parallel_reduce.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i, ValueType& update) const {
    update += i;
  }

  template <typename... Args>
  void test_reduce(const size_t work_size) {

  }

  // Run test_scan() for a collection of work size
  template <typename... Args>
  void test_reduce(const std::vector<size_t> work_sizes) {
    for (size_t i = 0; i < work_sizes.size(); ++i) {
      test_reduce<Args...>(work_sizes[i]);
    }
  }
};  // struct TestParallelReduceRangePolicy

TEST(TEST_CATEGORY, parallel_reduce_range_policy) {

}
}  // namespace
