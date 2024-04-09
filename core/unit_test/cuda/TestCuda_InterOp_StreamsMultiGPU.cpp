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
#include <TestMultiGPU.hpp>

namespace {

struct StreamsAndDevices {
  std::array<cudaStream_t, 2> streams;
  std::array<int, 2> devices;

  StreamsAndDevices() {
    int n_devices;
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDeviceCount(&n_devices));

    devices = {0, n_devices - 1};
    for (int i = 0; i < 2; ++i) {
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(devices[i]));
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&streams[i]));
    }
  }
  StreamsAndDevices(const StreamsAndDevices &) = delete;
  StreamsAndDevices &operator=(const StreamsAndDevices &) = delete;
  ~StreamsAndDevices() {
    for (int i = 0; i < 2; ++i) {
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(devices[i]));
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(streams[i]));
    }
  }
};

std::array<TEST_EXECSPACE, 2> get_execution_spaces(
    const StreamsAndDevices &streams_and_devices) {
  TEST_EXECSPACE exec0(streams_and_devices.streams[0]);
  TEST_EXECSPACE exec1(streams_and_devices.streams[1]);

  // Must return void to use ASSERT_EQ
  [&]() {
    ASSERT_EQ(exec0.cuda_device(), streams_and_devices.devices[0]);
    ASSERT_EQ(exec1.cuda_device(), streams_and_devices.devices[1]);
  }();

  return {exec0, exec1};
}

TEST(cuda_multi_gpu, managed_views) {
  StreamsAndDevices streams_and_devices;
  {
    std::array<TEST_EXECSPACE, 2> execs =
        get_execution_spaces(streams_and_devices);

    Kokkos::View<int *, TEST_EXECSPACE> view0(
        Kokkos::view_alloc("v0", execs[0]), 100);
    Kokkos::View<int *, TEST_EXECSPACE> view(Kokkos::view_alloc("v", execs[1]),
                                             100);

    test_policies(execs[0], view0, execs[1], view);
  }
}

TEST(cuda_multi_gpu, unmanaged_views) {
  StreamsAndDevices streams_and_devices;
  {
    std::array<TEST_EXECSPACE, 2> execs =
        get_execution_spaces(streams_and_devices);

    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(execs[0].cuda_device()));
    int *p0;
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaMalloc(reinterpret_cast<void **>(&p0), sizeof(int) * 100));
    Kokkos::View<int *, TEST_EXECSPACE> view0(p0, 100);

    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(execs[1].cuda_device()));
    int *p;
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaMalloc(reinterpret_cast<void **>(&p), sizeof(int) * 100));
    Kokkos::View<int *, TEST_EXECSPACE> view(p, 100);

    test_policies(execs[0], view0, execs[1], view);
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(p0));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(p));
  }
}

TEST(cuda_multi_gpu, scratch_space) {
  StreamsAndDevices streams_and_devices;
  {
    std::array<TEST_EXECSPACE, 2> execs =
        get_execution_spaces(streams_and_devices);

    test_scratch(execs[0], execs[1]);
  }
}





template <class MemSpace>
struct TestViewCudaAccessible {
  enum { N = 1000 };

  std::array<TEST_EXECSPACE, 2> execs;
  using V = Kokkos::View<int*, MemSpace>;

  V m_v;

  struct TagInit {};
  struct TagTest {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagInit &, const int i) const {
    m_v(i) = i + 1;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagTest &, const int i, int &error_count) const {
    if (m_v(i) != i + 1) ++error_count;
  }

  TestViewCudaAccessible(std::array<TEST_EXECSPACE, 2> execs_) :
      execs(execs_),
      m_v("v0", N) {}

  void run() {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<typename MemSpace::execution_space, TagInit>(0, N),
        *this);
    Kokkos::fence();

    // Next access is a different execution space, must complete prior kernel.
    int err0, err1;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<TEST_EXECSPACE, TagTest>(execs[0], 0, N), *this,
                            err0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<TEST_EXECSPACE, TagTest>(execs[1], 0, N), *this,
                            err1);
    EXPECT_EQ(err0, 0);
    EXPECT_EQ(err1, 0);
  }
};

TEST(cuda_multi_gpu, diff_mem_space) {
  StreamsAndDevices streams_and_devices;
  {
    std::array<TEST_EXECSPACE, 2> execs =
        get_execution_spaces(streams_and_devices);

    TestViewCudaAccessible<Kokkos::CudaUVMSpace> test_uvm(execs);
    test_uvm.run();

    TestViewCudaAccessible<Kokkos::CudaHostPinnedSpace> test_hp(execs);
    test_hp.run();
  }
}
}  // namespace
