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

template <class Exec, class X, class Y>
KOKKOS_INLINE_FUNCTION void sum_views(const Exec& exec, const X& x,
                                      const Y& y) {
  auto policy = Kokkos::MDRangePolicy(exec, {0, 0}, {x.extent(0), x.extent(1)});
  Kokkos::parallel_for(
      policy,
      KOKKOS_LAMBDA(const int& i, const int& j) { x(i, j) += y(i, j); });
}

void test_self_similar_mdrange_policy_computation() {
  using ViewType     = typename Kokkos::View<int***>;
  using HostViewType = typename ViewType::host_mirror_type;

  int dims[] = {3, 5, 7};

  int num_teams = dims[0];
  int n0        = dims[1];
  int n1        = dims[2];

  Kokkos::View<int**> v_x("v_x", n0, n1), v_y("v_y", n0, n1);
  Kokkos::View<int***> M_x("M_x", num_teams, n0, n1),
      M_y("M_y", num_teams, n0, n1);

  // Initialize v_x and v_y with values from 1 to N
  Kokkos::parallel_for(
      "init_v_x",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(Kokkos::DefaultExecutionSpace(),
                                             {0, 0}, {n0, n1}),
      KOKKOS_LAMBDA(const int& i, const int& j) {
        v_x(i, j) = i * n1 + j + 1;
      });
  Kokkos::parallel_for(
      "init_v_y",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(Kokkos::DefaultExecutionSpace(),
                                             {0, 0}, {n0, n1}),
      KOKKOS_LAMBDA(const int& i, const int& j) {
        v_y(i, j) = i * n1 + j + 1;
      });

  // Initialize M_x and M_y with values from 1 to M (flattened index)
  Kokkos::parallel_for(
      "init_M_x",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(Kokkos::DefaultExecutionSpace(),
                                             {0, 0, 0}, {num_teams, n0, n1}),
      KOKKOS_LAMBDA(const int& i, const int& j, const int& k) {
        M_x(i, j, k) = i * n0 * n1 + j * n1 + k + 1;
      });
  Kokkos::parallel_for(
      "init_M_y",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(Kokkos::DefaultExecutionSpace(),
                                             {0, 0, 0}, {num_teams, n0, n1}),
      KOKKOS_LAMBDA(const int& i, const int& j, const int& k) {
        M_y(i, j, k) = i * n0 * n1 + j * n1 + k + 1;
      });

  //   auto v_x_host = Kokkos::create_mirror_view_and_copy(
  //       Kokkos::DefaultHostExecutionSpace(), v_x);
  //   auto v_y_host = Kokkos::create_mirror_view_and_copy(
  //       Kokkos::DefaultHostExecutionSpace(), v_y);
  //   auto M_x_host = Kokkos::create_mirror_view_and_copy(
  //       Kokkos::DefaultHostExecutionSpace(), M_x);
  //   auto M_y_host = Kokkos::create_mirror_view_and_copy(
  //       Kokkos::DefaultHostExecutionSpace(), M_y);

  //   Kokkos::parallel_for(
  //       Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
  //       Kokkos::Rank<2>>(
  //           Kokkos::DefaultHostExecutionSpace(), {0, 0}, {n0, n1}),
  //       KOKKOS_LAMBDA(const int& i, const int& j) {
  //         printf(
  //             "v_x(%d, %d) = %d, v_y(%d, %d) = %d\n",
  //             i, j, v_x_host(i, j), i, j, v_y_host(i, j));
  //       });

  // Call sum_views(ExecSpace):
  sum_views(Kokkos::DefaultExecutionSpace(), v_x, v_y);

  // Call sum_views(TeamHandle)
  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "apxyFromTeam", Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        sum_views(team,
                  Kokkos::subview(M_x, team.league_rank(), Kokkos::ALL(),
                                  Kokkos::ALL()),
                  Kokkos::subview(M_y, team.league_rank(), Kokkos::ALL(),
                                  Kokkos::ALL()));
      });

  // Check v_x
  size_t result = 0;
  Kokkos::parallel_reduce(
      "Check1", v_x.extent(0) * v_x.extent(1),
      KOKKOS_LAMBDA(int i, size_t& val) {
        int i0 = i / v_x.extent(1);
        int i1 = i % v_x.extent(1);
        val += v_x(i0, i1);
      },
      result);
  auto N              = n0 * n1;
  size_t expected_v_x = N * (N + 1);
  ASSERT_EQ(result, expected_v_x);

  // Check individual elements of v_x
  Kokkos::parallel_reduce(
      "Check1_elements",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(Kokkos::DefaultExecutionSpace(),
                                             {0, 0}, {n0, n1}),
      KOKKOS_LAMBDA(int i, int j, size_t& errors) {
        auto expected = 2 * (i * n1 + j + 1);
        if (v_x(i, j) != expected) ++errors;
      },
      result);
  ASSERT_EQ(result, size_t(0));

  // // Check M_x
  // result = 0;
  // Kokkos::parallel_reduce(
  //     "Check2", M_x.extent(0),
  //     KOKKOS_LAMBDA(int i, size_t& val) {
  //       for (int j = 0; j < M_x.extent_int(1); j++) val += M_x(i, j);
  //     },
  //     result);
  // size_t M_total      = num_teams * N;
  // size_t expected_M_x = M_total * (M_total + 1);
  // ASSERT_EQ(result, expected_M_x);

  // // Check individual elements of M_x
  // Kokkos::parallel_reduce(
  //     "Check2_elements", M_x.extent(0),
  //     KOKKOS_LAMBDA(int i, size_t& errors) {
  //       for (int j = 0; j < M_x.extent_int(1); j++) {
  //         float expected = static_cast<float>(2 * (i * N + j + 1));
  //         if (M_x(i, j) != expected) ++errors;
  //       }
  //     },
  //     result);
  // ASSERT_EQ(result, size_t(0));
}

TEST(defaultdevicetype, development_test) {
  test_self_similar_mdrange_policy_computation();
}

}  // namespace Test
