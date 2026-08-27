#include "tests_tensor_fixtures.hpp"

TEST(MdLayout, SimplexPackedIsABijectionOnSortedMultiIndices) {
  using T = ddx::impl::md_tensor<double, ddx::impl::uniform_extents_t<4, 3>,
                                 ddx::impl::layout_simplex_packed>;
  EXPECT_EQ(T::size(), 20u); // C(4 + 3 - 1, 3) = C(6, 3) = 20

  constexpr T::mapping_type m{};
  std::set<std::size_t> slots;
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = i; j < 4; ++j) {
      for (std::size_t k = j; k < 4; ++k) {
        const auto s = static_cast<std::size_t>(m(i, j, k));
        EXPECT_LT(s, T::size());
        // Distinct multisets must not collide...
        EXPECT_TRUE(slots.insert(s).second);
        // ...and every permutation of one must.
        EXPECT_EQ(static_cast<std::size_t>(m(k, j, i)), s);
        EXPECT_EQ(static_cast<std::size_t>(m(j, i, k)), s);
        EXPECT_EQ(static_cast<std::size_t>(m(k, i, j)), s);
      }
    }
  }
  EXPECT_EQ(slots.size(), T::size()); // the packing wastes no cell
}
TEST(MdLayout, SparsePatternSendsStructuralZerosToTheSink) {
  // f = x*y + z*z couples {x,y} and {z,z}, but never x with z or y with z.
  ddx::impl::Variable<double, ddx::impl::FixedString{"x"}> x;
  ddx::impl::Variable<double, ddx::impl::FixedString{"y"}> y;
  ddx::impl::Variable<double, ddx::impl::FixedString{"z"}> z;
  using E = decltype(x * y + z * z);
  using L = ddx::impl::layout_sparse_pattern<E>;

  const typename L::template mapping<ddx::impl::md::extents<std::size_t, 3, 3>>
      m{};
  EXPECT_EQ(static_cast<std::size_t>(m.required_span_size()), L::kNnz + 1);

  // Symbols are alphabetical: x=0, y=1, z=2.
  EXPECT_TRUE(m.contains(0, 1));  // d2/dx dy is in the pattern
  EXPECT_TRUE(m.contains(2, 2));  // d2/dz2 too
  EXPECT_FALSE(m.contains(0, 2)); // x and z never meet
  EXPECT_FALSE(m.contains(1, 2));
  // Every structural zero shares the one sink cell.
  EXPECT_EQ(static_cast<std::size_t>(m(0, 2)), L::kNnz);
  EXPECT_EQ(static_cast<std::size_t>(m(1, 2)), L::kNnz);
  // ...and every nonzero has its own slot.
  EXPECT_NE(static_cast<std::size_t>(m(0, 1)),
            static_cast<std::size_t>(m(2, 2)));
}
TEST(MdTensor, BothIndexSpellingsAgree) {
  auto t = ddx::impl::nd_tensor_t<double, 3, 3>{};
  double v = 0.0;
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = i; j < 3; ++j) {
      for (std::size_t k = j; k < 3; ++k) {
        t[i, j, k] = (v += 1.0);
      }
    }
  }
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_DOUBLE_EQ((t[i, j, k]), t[i][j][k]);
      }
    }
  }
}
// The constexpr half of the contract: every mapping and the tensor itself must
// survive constant evaluation, or the symbolic path silently stops being
// constexpr and only the ConstexprContract suite would notice, much later.
TEST(ConstexprContract, MdspanLayerIsConstantEvaluated) {
  constexpr auto simplex = [] {
    ddx::impl::nd_tensor_t<double, 3, 3> t{};
    t[2, 1, 0] = 6.0;
    return (t[0, 1, 2]) + (t[1, 0, 2]);
  }();
  static_assert(simplex == 12.0);

  constexpr auto stacked = [] {
    ddx::impl::nd_stack_t<double, 2, 3, 2> s{};
    s[1, 0, 2] = 3.0;
    return (s[1, 2, 0]) + (s[0, 0, 2]); // second output set, first still zero
  }();
  static_assert(stacked == 3.0);

  SUCCEED(); // the static_asserts above are the test
}
