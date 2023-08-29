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

namespace {

template <typename Policy>
void test_run_time_parameters() {
  int league_size = 131;

  using ExecutionSpace = typename Policy::execution_space;
  int team_size =
      4 < ExecutionSpace().concurrency() ? 4 : ExecutionSpace().concurrency();
#ifdef KOKKOS_ENABLE_HPX
  team_size = 1;
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
  if (std::is_same<ExecutionSpace, Kokkos::Experimental::OpenMPTarget>::value)
    team_size = 32;
#endif
  int chunk_size         = 4;
  int per_team_scratch   = 1024;
  int per_thread_scratch = 16;
  int scratch_size       = per_team_scratch + per_thread_scratch * team_size;

  auto test_policy_attributes =
      [](const Policy p, int expected_league_size, int expected_team_size,
         int expected_chunk_size, size_t expected_scratch_size) {
        ASSERT_EQ(p.league_size(), expected_league_size);
        ASSERT_EQ(p.team_size(), expected_team_size);
        ASSERT_EQ(p.chunk_size(), expected_chunk_size);
        ASSERT_EQ(p.scratch_size(0), expected_scratch_size);
      };

  Policy p1(league_size, team_size);
  // chunk_size will vary by architecture, so do not check
  test_policy_attributes(p1, league_size, team_size, p1.chunk_size(), 0);
  ASSERT_GT(p1.chunk_size(), 0);

  Policy p2 = p1.set_chunk_size(chunk_size);
  test_policy_attributes(p1, league_size, team_size, chunk_size, 0);
  test_policy_attributes(p2, league_size, team_size, chunk_size, 0);

  Policy p3 = p2.set_scratch_size(0, Kokkos::PerTeam(per_team_scratch));
  test_policy_attributes(p2, league_size, team_size, chunk_size,
                         per_team_scratch);
  test_policy_attributes(p3, league_size, team_size, chunk_size,
                         per_team_scratch);

  Policy p4 = p2.set_scratch_size(0, Kokkos::PerThread(per_thread_scratch));
  test_policy_attributes(p2, league_size, team_size, chunk_size, scratch_size);
  test_policy_attributes(p4, league_size, team_size, chunk_size, scratch_size);

  Policy p5 = p2.set_scratch_size(0, Kokkos::PerThread(per_thread_scratch),
                                  Kokkos::PerTeam(per_team_scratch));
  test_policy_attributes(p2, league_size, team_size, chunk_size, scratch_size);
  test_policy_attributes(p5, league_size, team_size, chunk_size, scratch_size);

  Policy p6 = p2.set_scratch_size(0, Kokkos::PerTeam(per_team_scratch),
                                  Kokkos::PerThread(per_thread_scratch));
  test_policy_attributes(p2, league_size, team_size, chunk_size, scratch_size);
  test_policy_attributes(p6, league_size, team_size, chunk_size, scratch_size);

  Policy p7;  // default constructed
  // team_size and chunk_size will vary between architecture, so do not check
  test_policy_attributes(p7, 0, p7.team_size(), p7.chunk_size(), 0);

  p7 = p3;  // call assignment operator
  test_policy_attributes(p3, league_size, team_size, chunk_size, scratch_size);
  test_policy_attributes(p7, league_size, team_size, chunk_size, scratch_size);
}

TEST(TEST_CATEGORY, team_policy_runtime_parameters) {
  struct SomeTag {};

  using TestExecSpace   = TEST_EXECSPACE;
  using DynamicSchedule = Kokkos::Schedule<Kokkos::Dynamic>;
  using LongIndex       = Kokkos::IndexType<long>;

  // clang-format off
  test_run_time_parameters<Kokkos::TeamPolicy<TestExecSpace                                             >>();
  test_run_time_parameters<Kokkos::TeamPolicy<TestExecSpace,   DynamicSchedule, LongIndex               >>();
  test_run_time_parameters<Kokkos::TeamPolicy<LongIndex,       TestExecSpace,   DynamicSchedule         >>();
  test_run_time_parameters<Kokkos::TeamPolicy<DynamicSchedule, LongIndex,       TestExecSpace,   SomeTag>>();
  // clang-format on
}

}  // namespace
