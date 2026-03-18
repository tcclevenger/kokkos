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

struct my_custom_layout_right {
  template <class Extents>
  class mapping;
};

template <class Extents>
class my_custom_layout_right::mapping {
 public:
  using extents_type = Extents;
  using index_type   = typename extents_type::index_type;
  using size_type    = typename extents_type::size_type;
  using rank_type    = typename extents_type::rank_type;
  using layout_type  = my_custom_layout_right;

  KOKKOS_DEFAULTED_FUNCTION constexpr mapping(const mapping&) noexcept =
      default;

  KOKKOS_INLINE_FUNCTION
  constexpr mapping(const extents_type& ext,
                    const index_type stride_0_factor = 1)
      : m_extents(ext), m_stride_0_factor(stride_0_factor) {
    assert(m_stride_0_factor > 1);
    assert(extents_type::rank() > 1 || m_stride_0_factor == 1);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr const extents_type& extents() const noexcept { return m_extents; }

  KOKKOS_FUNCTION
  template <class... Indices>
  constexpr index_type operator()(Indices... idxs) const noexcept {
    static_assert(sizeof...(Indices) == extents_type::rank(),
                  "number of indices must match rank of extents");

    static_assert((std::is_convertible_v<Indices, index_type> && ...),
                  "indices must be integral types");

    index_type idx_vec[] = {static_cast<index_type>(idxs)...};

    index_type offset = 0;
    for (rank_type r = 0; r < extents_type::rank(); ++r) {
      offset += idx_vec[r] * stride(r);
    }
    return offset;
  }

  KOKKOS_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept {
    return true;
  }
  KOKKOS_INLINE_FUNCTION static constexpr bool is_always_exhaustive() noexcept {
    return false;
  }
  KOKKOS_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept {
    return true;
  }

  KOKKOS_INLINE_FUNCTION static constexpr bool is_unique() noexcept {
    return true;
  }
  KOKKOS_INLINE_FUNCTION bool is_exhaustive() const noexcept {
    return (extents_type::rank() < 2) || (m_stride_0_factor == 1);
  }
  KOKKOS_INLINE_FUNCTION static constexpr bool is_strided() noexcept {
    return true;
  }

  KOKKOS_INLINE_FUNCTION constexpr index_type stride(
      rank_type i) const noexcept {
    assert(i < extents_type::rank());

    if constexpr (extents_type::rank() == 0) return 0;
    if constexpr (extents_type::rank() == 1) return 1;

    index_type value = 1;
    for (rank_type r = extents_type::rank() - 1; r > i; r--) {
      value *= m_extents.extent(r);
    }
    if (i == 0) value *= m_stride_0_factor;
    return value;
  }

  KOKKOS_INLINE_FUNCTION
  constexpr index_type required_span_size() const noexcept {
    for (rank_type r = 0; r < extents_type::rank(); ++r) {
      if (m_extents.extent(r) == 0) return 0;
    }

    index_type n = 1;
    for (rank_type r = 0; r < extents_type::rank(); ++r) {
      n += (m_extents.extent(r) - 1) * stride(r);
    }
    return n;
  }

  template <class OtherExtents>
    requires(Extents::rank() == OtherExtents::rank())
  KOKKOS_INLINE_FUNCTION friend constexpr bool operator==(
      mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept {
    return lhs.extents() == rhs.extents() &&
           lhs.m_stride_0_factor == rhs.m_stride_0_factor;
  }

 private:
  extents_type m_extents{};
  index_type m_stride_0_factor = 1;

  template <class... SliceSpecifiers>
  KOKKOS_INLINE_FUNCTION constexpr auto submdspan_mapping_impl(
      SliceSpecifiers... slices) const {
    // compute sub extents
    using src_ext_t = Extents;
    auto dst_ext    = Kokkos::submdspan_extents(extents(), slices...);
    using dst_ext_t = decltype(dst_ext);
  }

  template <class... SliceSpecifiers>
  KOKKOS_INLINE_FUNCTION friend constexpr auto submdspan_mapping(
      const mapping& src, SliceSpecifiers... slices) {
    return src.submdspan_mapping_impl(slices...);
  }
};

TEST(defaultdevicetype, development_test) {}

}  // namespace Test
