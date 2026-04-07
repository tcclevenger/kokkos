// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

namespace Test {

// template <class Policy>
// KOKKOS_INLINE_FUNCTION int check_runtime_inputs(
//     Policy& p, const typename Policy::index_type expected_begin,
//     const typename Policy::index_type expected_end,
//     const typename Policy::index_type chunk_size = 1) {
//   int nerrs = 0;

//   if (p.begin() != expected_begin) ++nerrs;
//   if (p.end() != expected_end) ++nerrs;

//   auto p2 = p.set_chunk_size(chunk_size);
//   if (p2.chunk_size() != chunk_size) ++nerrs;

//   return nerrs;
// }

// void test_self_similar_mdrange_policy_runtime() {
//   using IndexType = typename Kokkos::DefaultExecutionSpace::size_type;

//   IndexType beg        = 5;
//   IndexType end        = 15;
//   IndexType chunk_size = 10;

//   auto p_execspace =
//       Kokkos::MDRangePolicy(Kokkos::DefaultExecutionSpace(), beg, end);
//   auto nerrs_exec_space =
//       check_runtime_inputs(p_execspace, beg, end, chunk_size);
//   ASSERT_EQ(nerrs_exec_space, 0);

//   int nerrs_team_handle;
//   using team_t = typename Kokkos::TeamPolicy<>::member_type;
//   Kokkos::parallel_reduce(
//       "check_runtime", Kokkos::TeamPolicy(1, Kokkos::AUTO()),
//       KOKKOS_LAMBDA(const team_t& team, int& nerrs) {
//         auto p_teamhandle = Kokkos::RangePolicy(team, beg, end);
//         auto tvr          = Kokkos::TeamVectorRange(team, beg, end);
//         nerrs = check_runtime_inputs(p_teamhandle, tvr.start, tvr.end);
//       },
//       nerrs_team_handle);
//   ASSERT_EQ(nerrs_team_handle, 0);
// }

template <class Exec, class X, class Y>
KOKKOS_INLINE_FUNCTION void sum_views(const Exec& exec, const X& x,
                                      const Y& y) {
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(exec, {0, 0}, {x.extent(0), x.extent(1)});
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const int& i, const int& j) { x(i, j) += y(i, j); });
}

void test_self_similar_mdrange_policy_computation() {
  using ViewType     = typename Kokkos::View<int***>;
  using HostViewType = typename ViewType::host_mirror_type;

  int num_teams = dims[0];
  int n0 = dims[1];
  int n1 = dims[2];

  Kokkos::View<int**> v_x("v_x", n0, n1), v_y("v_y", n0, n1);
  Kokkos::View<int***> M_x("M_x", num_teams, n0, n1), M_y("M_y", num_teams, n0, n1);

  // Initialize v_x and v_y with values from 1 to N
  Kokkos::parallel_for(
      "init_v_x", Kokkos::MDRangePolicy<Kokkos::Rank<2>>(Kokkos::DefaultExecutionSpace(), {0, 0}, {n0, n1}),
      KOKKOS_LAMBDA(const int& i, const int& j) { v_x(i, j) = i * n1 + j + 1; });
  Kokkos::parallel_for(
      "init_v_y", Kokkos::MDRangePolicy<Kokkos::Rank<2>>(Kokkos::DefaultExecutionSpace(), {0, 0}, {n0, n1}),
      KOKKOS_LAMBDA(const int& i, const int& j) { v_y(i, j) = i * n1 + j + 1; });

  // Initialize M_x and M_y with values from 1 to M (flattened index)
  Kokkos::parallel_for(
      "init_M_x", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(Kokkos::DefaultExecutionSpace(), {0, 0, 0}, {num_teams, n0, n1}),
      KOKKOS_LAMBDA(const int& i, const int& j, const int& k) {
          M_x(i, j, k) = i * n0 * n1 + j * n1 + k + 1;
      });
  Kokkos::parallel_for(
      "init_M_y", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(Kokkos::DefaultExecutionSpace(), {0, 0, 0}, {num_teams, n0, n1}),
      KOKKOS_LAMBDA(const int& i, const int& j, const int& k) {
          M_y(i, j, k) = i * n0 * n1 + j * n1 + k + 1;
      });

  // // Call sum_views(ExecSpace):
  // sum_views(Kokkos::DefaultExecutionSpace(), v_x, v_y);

  // // Call sum_views(TeamHandle)
  // using team_t = typename Kokkos::TeamPolicy<>::member_type;
  // Kokkos::parallel_for(
  //     "apxyFromTeam", Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()),
  //     KOKKOS_LAMBDA(const team_t& team) {
  //       sum_views(team, Kokkos::subview(M_x, team.league_rank(), Kokkos::ALL()),
  //                 Kokkos::subview(M_y, team.league_rank(), Kokkos::ALL()));
  //     });

  // // Check v_x
  // size_t result = 0;
  // Kokkos::parallel_reduce(
  //     "Check1", v_x.extent(0),
  //     KOKKOS_LAMBDA(int i, size_t& val) { val += v_x(i); }, result);
  // size_t expected_v_x = N * (N + 1);
  // ASSERT_EQ(result, expected_v_x);

  // // Check individual elements of v_x
  // Kokkos::parallel_reduce(
  //     "Check1_elements", v_x.extent(0),
  //     KOKKOS_LAMBDA(int i, size_t& errors) {
  //       float expected = static_cast<float>(2 * (i + 1));
  //       if (v_x(i) != expected) ++errors;
  //     },
  //     result);
  // ASSERT_EQ(result, size_t(0));

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

// TEST(TEST_CATEGORY, self_similar_mdrange_policy_runtime) {
//   test_self_similar_mdrange_policy_runtime();
// }

TEST(TEST_CATEGORY, self_similar_mdrange_policy_computation) {
  test_self_similar_mdrange_policy_computation();
}

}  // namespace Test
