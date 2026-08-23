// Boost's -fno-exceptions protocol.  Under BOOST_NO_EXCEPTIONS -- which Boost
// sets for itself the moment it sees the flag -- boost::throw_exception is
// declared but not defined, and the program is required to supply it.  Nothing
// in ddx routes through it: it is reached only from Boost's own internals
// (any_cast in the property maps, shared_count, the graph containers), where
// the alternative to terminating is a container in a state Boost will not
// describe.
//
// Compiled into libddx only when DDX_NO_EXCEPTIONS is on; with exceptions
// enabled Boost defines these itself and this file is not built.
//
// Both carry default visibility explicitly.  libddx is built with
// -fvisibility=hidden, and these are the one pair of definitions in it that has
// to be found by Boost's own inline code compiled into other objects -- the JIT
// half of the library included.  They are in namespace boost, so DDX_API is not
// theirs to wear.
#include <boost/assert/source_location.hpp>
#include <boost/throw_exception.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace boost {

#if defined(__GNUC__) || defined(__clang__)
#define DDX_BOOST_THROW_VISIBILITY __attribute__((visibility("default")))
#else
#define DDX_BOOST_THROW_VISIBILITY
#endif

DDX_BOOST_THROW_VISIBILITY void throw_exception(const std::exception &e) {
  std::fprintf(stderr, "ddx: boost threw with exceptions disabled: %s\n",
               e.what());
  std::abort();
}

DDX_BOOST_THROW_VISIBILITY void
throw_exception(const std::exception &e, const source_location &loc) {
  std::fprintf(stderr,
               "ddx: boost threw with exceptions disabled: %s (%s:%u)\n",
               e.what(), loc.file_name(), loc.line());
  std::abort();
}

} // namespace boost
