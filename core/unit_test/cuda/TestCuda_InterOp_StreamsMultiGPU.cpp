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

#include <TestCuda_Category.hpp>
#include <Test_InterOp_Streams.hpp>

namespace Test {
// Test Interoperability with Cuda Streams
TEST(cuda, raw_cuda_streams) {
  Kokkos::initialize();

  int n_devices;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDeviceCount(&n_devices));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(0));
  cudaStream_t stream0;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream0));
  int *p0;
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMalloc(reinterpret_cast<void **>(&p0), sizeof(int) * 100));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(n_devices - 1));
  cudaStream_t stream;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream));
  int *p;
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMalloc(reinterpret_cast<void **>(&p), sizeof(int) * 100));
  using MemorySpace = typename TEST_EXECSPACE::memory_space;

  {
    TEST_EXECSPACE space0(0, stream0);
    Kokkos::View<int *, TEST_EXECSPACE> v0(p0, 100);
    Kokkos::deep_copy(space0, v0, 5);
    TEST_EXECSPACE space(n_devices - 1, stream);
    Kokkos::View<int *, TEST_EXECSPACE> v(p, 100);
    Kokkos::deep_copy(space, v, 5);

    int sum;
    int sum0;

    Kokkos::parallel_for("Test::cuda::raw_cuda_stream::Range_0",
                         Kokkos::RangePolicy<TEST_EXECSPACE>(space0, 0, 100),
                         FunctorRange<MemorySpace>(v0));
    Kokkos::parallel_for("Test::cuda::raw_cuda_stream::Range",
                         Kokkos::RangePolicy<TEST_EXECSPACE>(space, 0, 100),
                         FunctorRange<MemorySpace>(v));
    Kokkos::parallel_reduce(
        "Test::cuda::raw_cuda_stream::RangeReduce_0",
        Kokkos::RangePolicy<TEST_EXECSPACE, Kokkos::LaunchBounds<128, 2>>(
            space0, 0, 100),
        FunctorRangeReduce<MemorySpace>(v0), sum0);
    Kokkos::parallel_reduce(
        "Test::cuda::raw_cuda_stream::RangeReduce",
        Kokkos::RangePolicy<TEST_EXECSPACE, Kokkos::LaunchBounds<128, 2>>(
            space, 0, 100),
        FunctorRangeReduce<MemorySpace>(v), sum);
    ASSERT_EQ(600, sum0);
    ASSERT_EQ(600, sum);

    Kokkos::parallel_for("Test::cuda::raw_cuda_stream::MDRange_0",
                         Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>(
                             space0, {0, 0}, {10, 10}),
                         FunctorMDRange<MemorySpace>(v0));
    Kokkos::parallel_for("Test::cuda::raw_cuda_stream::MDRange",
                         Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>(
                             space, {0, 0}, {10, 10}),
                         FunctorMDRange<MemorySpace>(v));
    Kokkos::parallel_reduce(
        "Test::cuda::raw_cuda_stream::MDRangeReduce_0",
        Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                              Kokkos::LaunchBounds<128, 2>>(space0, {0, 0},
                                                            {10, 10}),
        FunctorMDRangeReduce<MemorySpace>(v0), sum0);
    Kokkos::parallel_reduce(
        "Test::cuda::raw_cuda_stream::MDRangeReduce",
        Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                              Kokkos::LaunchBounds<128, 2>>(space, {0, 0},
                                                            {10, 10}),
        FunctorMDRangeReduce<MemorySpace>(v), sum);
    ASSERT_EQ(700, sum0);
    ASSERT_EQ(700, sum);

    Kokkos::parallel_for("Test::cuda::raw_cuda_stream::Team_0",
                         Kokkos::TeamPolicy<TEST_EXECSPACE>(space0, 10, 10),
                         FunctorTeam<MemorySpace, TEST_EXECSPACE>(v0));
    Kokkos::parallel_for("Test::cuda::raw_cuda_stream::Team",
                         Kokkos::TeamPolicy<TEST_EXECSPACE>(space, 10, 10),
                         FunctorTeam<MemorySpace, TEST_EXECSPACE>(v));
    Kokkos::parallel_reduce(
        "Test::cuda::raw_cuda_stream::Team_0",
        Kokkos::TeamPolicy<TEST_EXECSPACE, Kokkos::LaunchBounds<128, 2>>(
            space0, 10, 10),
        FunctorTeamReduce<MemorySpace, TEST_EXECSPACE>(v0), sum0);
    Kokkos::parallel_reduce(
        "Test::cuda::raw_cuda_stream::Team",
        Kokkos::TeamPolicy<TEST_EXECSPACE, Kokkos::LaunchBounds<128, 2>>(
            space, 10, 10),
        FunctorTeamReduce<MemorySpace, TEST_EXECSPACE>(v), sum);
    ASSERT_EQ(800, sum0);
    ASSERT_EQ(800, sum);
  }
  Kokkos::finalize();

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(p0));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(p));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(0));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(stream0));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(n_devices - 1));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(stream));
}
}  // namespace Test
