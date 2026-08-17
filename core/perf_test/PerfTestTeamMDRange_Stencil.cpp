// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <iostream>
#include <limits>

#include <benchmark/benchmark.h>

#include "PerfTest_Category.hpp"
#include <Kokkos_Core.hpp>

namespace Test {

template <typename Layout>
struct LayoutToIterationPattern {};

template <>
struct LayoutToIterationPattern<Kokkos::LayoutRight> {
  static constexpr Kokkos::Iterate pattern = Kokkos::Iterate::Right;
};

template <>
struct LayoutToIterationPattern<Kokkos::LayoutLeft> {
  static constexpr Kokkos::Iterate pattern = Kokkos::Iterate::Left;
};

template <typename ScalarType, typename ViewType>
void check_computation_3d_stencil(const ViewType& A, const ViewType& B,
                                  int league, int n0, int n1, int n2) {
  int num_errors = 0;
  auto Ahost     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  auto Bhost     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B);

  ScalarType epsilon = std::numeric_limits<ScalarType>::epsilon() * 100;
  for (int l = 0; l < league; ++l) {
    for (int i = 1; i < n0 + 1; ++i) {
      for (int j = 1; j < n1 + 1; ++j) {
        for (int k = 1; k < n2 + 1; ++k) {
          ScalarType check =
              0.25 *
              (ScalarType)(Bhost(l, i + 1, j, k) + Bhost(l, i - 1, j, k) +
                           Bhost(l, i, j + 1, k) + Bhost(l, i, j - 1, k) +
                           Bhost(l, i, j, k + 1) + Bhost(l, i, j, k - 1) +
                           Bhost(l, i, j, k));
          if (Kokkos::abs(Ahost(l, i, j, k) - check) > epsilon) {
            ++num_errors;
            std::cerr << "Correctness error at index: " << l << "," << i
                      << "," << j << "," << k << ", got "
                      << Ahost(l, i, j, k) << ", expected " << check << "\n";
          }
        }
      }
    }
  }

  if (num_errors != 0) {
    std::cerr << "Detected " << num_errors
              << " errors in Team*MDRange 3D stencil benchmark"
              << std::endl;
  }
}

template <class DeviceType, typename TestLayout = Kokkos::LayoutRight,
          typename ScalarType = double>
struct TeamThreadMDRangeStencil {
  using execution_space = DeviceType;
  using scalar_type     = ScalarType;
  using team_policy     = Kokkos::TeamPolicy<execution_space>;
  using team_member     = typename team_policy::member_type;
  using view_type       = Kokkos::View<ScalarType****, TestLayout, DeviceType>;

  static constexpr int dimension = 4;

  view_type A;
  view_type B;
  const int m_n0;
  const int m_n1;
  const int m_n2;

  TeamThreadMDRangeStencil(const view_type& A_, const view_type& B_, int n0,
                           int n1, int n2)
      : A(A_), B(B_), m_n0(n0), m_n1(n1), m_n2(n2) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& team) const {
    constexpr Kokkos::Iterate iteration_pattern =
        LayoutToIterationPattern<TestLayout>::pattern;
    using rank_type = Kokkos::Rank<3, iteration_pattern, iteration_pattern>;

    const int league_rank = team.league_rank();
    auto team_range =
        Kokkos::TeamThreadMDRange<rank_type, team_member>(team, m_n0, m_n1, m_n2);

    Kokkos::parallel_for(team_range, [=, this](int i0, int i1, int i2) {
      i0++;
      i1++;
      i2++;
      A(league_rank, i0, i1, i2) =
          0.25 *
          (ScalarType)(B(league_rank, i0 + 1, i1, i2) + B(league_rank, i0 - 1, i1, i2) +
                       B(league_rank, i0, i1 + 1, i2) + B(league_rank, i0, i1 - 1, i2) +
                       B(league_rank, i0, i1, i2 + 1) + B(league_rank, i0, i1, i2 - 1) +
                       B(league_rank, i0, i1, i2));
    });
  }

  static auto get_policy(int league_size) {
    return team_policy(league_size, Kokkos::AUTO);
  }
};

template <class DeviceType, typename TestLayout = Kokkos::LayoutRight,
          typename ScalarType = double>
struct TeamVectorMDRangeStencil {
  using execution_space = DeviceType;
  using scalar_type     = ScalarType;
  using team_policy     = Kokkos::TeamPolicy<execution_space>;
  using team_member     = typename team_policy::member_type;
  using view_type       = Kokkos::View<ScalarType****, TestLayout, DeviceType>;

  static constexpr int dimension = 4;

  view_type A;
  view_type B;
  const int m_n0;
  const int m_n1;
  const int m_n2;

  TeamVectorMDRangeStencil(const view_type& A_, const view_type& B_, int n0,
                           int n1, int n2)
      : A(A_), B(B_), m_n0(n0), m_n1(n1), m_n2(n2) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& team) const {
    constexpr Kokkos::Iterate iteration_pattern =
        LayoutToIterationPattern<TestLayout>::pattern;
    using rank_type = Kokkos::Rank<3, iteration_pattern, iteration_pattern>;

    const int league_rank = team.league_rank();
    auto team_range =
        Kokkos::TeamVectorMDRange<rank_type, team_member>(team, m_n0, m_n1, m_n2);

    Kokkos::parallel_for(team_range, [=, this](int i0, int i1, int i2) {
      i0++;
      i1++;
      i2++;
      A(league_rank, i0, i1, i2) =
          0.25 *
          (ScalarType)(B(league_rank, i0 + 1, i1, i2) + B(league_rank, i0 - 1, i1, i2) +
                       B(league_rank, i0, i1 + 1, i2) + B(league_rank, i0, i1 - 1, i2) +
                       B(league_rank, i0, i1, i2 + 1) + B(league_rank, i0, i1, i2 - 1) +
                       B(league_rank, i0, i1, i2));
    });
  }

  static auto get_policy(int league_size) {
    return team_policy(league_size, Kokkos::AUTO,
                       team_policy::vector_length_max());
  }
};

template <class DeviceType, typename TestLayout = Kokkos::LayoutRight,
          typename ScalarType = double>
struct ThreadVectorMDRangeStencil {
  using execution_space = DeviceType;
  using scalar_type     = ScalarType;
  using team_policy     = Kokkos::TeamPolicy<execution_space>;
  using team_member     = typename team_policy::member_type;
  using view_type = Kokkos::View<ScalarType****, TestLayout, DeviceType>;

  static constexpr int dimension = 4;

  view_type A;
  view_type B;
  const int m_n0;
  const int m_n1;
  const int m_n2;

  ThreadVectorMDRangeStencil(const view_type& A_, const view_type& B_, int n0,
                             int n1, int n2)
      : A(A_), B(B_), m_n0(n0), m_n1(n1), m_n2(n2) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& team) const {
    constexpr Kokkos::Iterate iteration_pattern =
        LayoutToIterationPattern<TestLayout>::pattern;
    using rank_type = Kokkos::Rank<2, iteration_pattern, iteration_pattern>;

    const int league_rank = team.league_rank();
    auto team_thread_range = Kokkos::TeamThreadRange(team, m_n0);

    Kokkos::parallel_for(team_thread_range, [=, this](int i0) {
      const int i = i0 + 1;
      auto vector_md_range =
          Kokkos::ThreadVectorMDRange<rank_type, team_member>(team, m_n1, m_n2);

      Kokkos::parallel_for(vector_md_range, [=, this](int i1, int i2) {
        i1++;
        i2++;
        A(league_rank, i, i1, i2) =
            0.25 *
            (ScalarType)(B(league_rank, i + 1, i1, i2) +
                         B(league_rank, i - 1, i1, i2) +
                         B(league_rank, i, i1 + 1, i2) +
                         B(league_rank, i, i1 - 1, i2) +
                         B(league_rank, i, i1, i2 + 1) +
                         B(league_rank, i, i1, i2 - 1) +
                         B(league_rank, i, i1, i2));
      });
    });
  }

  static auto get_policy(int league_size) {
    return team_policy(league_size, Kokkos::AUTO,
                       team_policy::vector_length_max());
  }
};

template <typename FunctorType>
void bench_team_mdrange_3d(benchmark::State& state) {
  using execution_space = typename FunctorType::execution_space;
  using view_type       = typename FunctorType::view_type;

  const int league_size = static_cast<int>(state.range(0));
  const int size        = static_cast<int>(state.range(1));

  state.counters["league_size"] = league_size;
  state.counters["size"]        = size;

  view_type Atest("Atest", league_size, size + 2, size + 2, size + 2);
  view_type Btest("Btest", league_size, size + 2, size + 2, size + 2);

  Kokkos::deep_copy(Atest, 1.0);
  execution_space().fence();
  Kokkos::deep_copy(Btest, 1.0);
  execution_space().fence();

  const auto policy = FunctorType::get_policy(league_size);

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::parallel_for(policy, FunctorType(Atest, Btest, size, size, size));
    execution_space().fence();
    const double dt = timer.seconds();
    state.SetIterationTime(dt);
  }

  check_computation_3d_stencil<typename FunctorType::scalar_type>(
      Atest, Btest, league_size, size, size, size);
}

#if !defined(KOKKOS_ENABLE_BENCHMARKS_HEAVY)
#define TEAM_MDRANGE_STENCIL_BENCHMARK(functor, layout, fn, ...)          \
  BENCHMARK(fn<functor<TEST_EXECSPACE, Kokkos::layout>>)                  \
      ->UseManualTime()                                                    \
      ->Unit(benchmark::kMillisecond)                                      \
      ->Name("TeamMDRangeStencil_" #functor "_" #layout)                 \
      ->ArgNames({"league_size", "size"})                                \
      ->ArgsProduct({__VA_ARGS__})                                         \
      ->Iterations(1);

TEAM_MDRANGE_STENCIL_BENCHMARK(TeamThreadMDRangeStencil, LayoutRight,
                               bench_team_mdrange_3d, {16}, {48})
TEAM_MDRANGE_STENCIL_BENCHMARK(TeamVectorMDRangeStencil, LayoutRight,
                               bench_team_mdrange_3d, {16}, {48})
TEAM_MDRANGE_STENCIL_BENCHMARK(ThreadVectorMDRangeStencil, LayoutRight,
                               bench_team_mdrange_3d, {16}, {48})
#else
#define TEAM_MDRANGE_STENCIL_BENCHMARK(functor, layout, fn, ...)          \
  BENCHMARK(fn<functor<TEST_EXECSPACE, Kokkos::layout>>)                  \
      ->UseManualTime()                                                    \
      ->Unit(benchmark::kMillisecond)                                      \
      ->Name("TeamMDRangeStencil_" #functor "_" #layout)                 \
      ->ArgNames({"league_size", "size"})                                \
      ->ArgsProduct({__VA_ARGS__});

#define LEAGUE_SIZES \
  { 32, 64, 96 }
#define SIZES_3D \
  { 32, 64, 96 }

TEAM_MDRANGE_STENCIL_BENCHMARK(TeamThreadMDRangeStencil, LayoutRight,
                               bench_team_mdrange_3d, LEAGUE_SIZES, SIZES_3D)
TEAM_MDRANGE_STENCIL_BENCHMARK(TeamThreadMDRangeStencil, LayoutLeft,
                               bench_team_mdrange_3d, LEAGUE_SIZES, SIZES_3D)
TEAM_MDRANGE_STENCIL_BENCHMARK(TeamVectorMDRangeStencil, LayoutRight,
                               bench_team_mdrange_3d, LEAGUE_SIZES, SIZES_3D)
TEAM_MDRANGE_STENCIL_BENCHMARK(TeamVectorMDRangeStencil, LayoutLeft,
                               bench_team_mdrange_3d, LEAGUE_SIZES, SIZES_3D)
TEAM_MDRANGE_STENCIL_BENCHMARK(ThreadVectorMDRangeStencil, LayoutRight,
                               bench_team_mdrange_3d, LEAGUE_SIZES, \
                               SIZES_3D)
TEAM_MDRANGE_STENCIL_BENCHMARK(ThreadVectorMDRangeStencil, LayoutLeft,
                               bench_team_mdrange_3d, LEAGUE_SIZES, \
                               SIZES_3D)
#undef LEAGUE_SIZES
#undef SIZES_3D
#endif

#undef TEAM_MDRANGE_STENCIL_BENCHMARK

}  // namespace Test
