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

template <size_t Padding>
struct layout_right_padded_1 {
  template <class Extents>
  class mapping;
};

template <size_t Padding>
template <class Extents>
class layout_right_padded_1<Padding>::mapping {
 public:
  using extents_type = Extents;
  using index_type   = typename extents_type::index_type;
  using size_type    = typename extents_type::size_type;
  using rank_type    = typename extents_type::rank_type;
  using layout_type  = layout_right;

  KOKKOS_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept {
    return true;
  }
  KOKKOS_INLINE_FUNCTION static constexpr bool is_always_exhaustive() noexcept {
    return (extents_type::rank() <= rank_type(1)) ||
           (extents_type::static_extent(1) != Kokkos::dynamic_extent &&
            extents_type::static_extent(1) == 0);
  }
  KOKKOS_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept {
    return true;
  }

  KOKKOS_INLINE_FUNCTION static constexpr bool is_unique() noexcept {
    return true;
  }
  KOKKOS_INLINE_FUNCTION constexpr bool is_exhaustive() const noexcept {
    return (extents_type::rank() < 2) || (exts.extent(1) == m_padded_stride);
  }
  KOKKOS_INLINE_FUNCTION static constexpr bool is_strided() noexcept {
    return true;
  }

  KOKKOS_INLINE_FUNCTION constexpr index_type stride(
      rank_type r) const noexcept {
    assert(r < extents_type::rank());
    if (r == extents_type::rank() - 1) return index_type(1);

    index_type value = 1;
    for (rank_type k = extents_type::rank() - 2; k > r; k--)
      value *= m_extents.extent(k);
    if (r == 0) value *= m_padded_stride;

    return value;
  }

  KOKKOS_INLINE_FUNCTION
  template <class OtherExtents>
    requires Extents::rank() ==
  OtherExtents::rank() friend constexpr bool operator==(
      mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept {
    return lhs.extents() == rhs.extents();
  }

 private:
  extents_type m_extents{};
  size_t m_padded_stride;
};

TEST(defaultdevicetype, development_test) {}

}  // namespace Test
