// Under BOOST_NO_EXCEPTIONS boost::throw_exception is declared but not
// defined, and the program must supply it.  Nothing in ddx routes through it:
// it is reached only from Boost's own internals, where the alternative to
// terminating is a container in a state Boost will not describe.  Compiled
// into libddx unconditionally: the project builds without exceptions.
//
// Default visibility explicitly: libddx is -fvisibility=hidden, and Boost's
// inline code in other objects has to find these.  They live in namespace
// boost, so DDX_API is not theirs to wear.
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

DDX_BOOST_THROW_VISIBILITY void throw_exception(const std::exception &e,
                                                const source_location &loc) {
  std::fprintf(stderr,
               "ddx: boost threw with exceptions disabled: %s (%s:%u)\n",
               e.what(), loc.file_name(), loc.line());
  std::abort();
}

} // namespace boost
