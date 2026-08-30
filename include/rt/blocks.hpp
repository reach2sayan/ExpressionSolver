#pragma once

#include <boost/describe/class.hpp>

#include <concepts>
#include <cstddef>

namespace ddx::rt {

// One of a thing per output block, in the order the kernel writes them.  The
// ids a builder collects, the spans a frozen graph lends, the column counts
// read off them and the columns a kernel loads are the same three names, so
// they are said once.  Here rather than in graph.hpp because jit/kernel.hpp
// wants the counts and deliberately does not include a Graph.
template <std::semiregular Per> struct Blocks {
  Per values{};
  Per jacobian{};
  Per hessian{};
  BOOST_DESCRIBE_CLASS(Blocks, (), (values, jacobian, hessian), (), ())
};

// The column counts, read off the blocks so no column can be miscounted:
// `values` is m, `jacobian` is the pattern's nonzeros row-major by function,
// and `hessian` is colours * n, compressed.
using Layout = Blocks<std::size_t>;

} // namespace ddx::rt
