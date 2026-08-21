// Boost's -fno-exceptions protocol.  Under BOOST_NO_EXCEPTIONS -- which Boost
// sets for itself the moment it sees the flag -- boost::throw_exception is
// declared but not defined, and the program is required to supply it.  Nothing
// in ddx routes through it: it is reached only from Boost's own internals
// (any_cast in the property maps, shared_count, the graph containers), where
// the alternative to terminating is a container in a state Boost will not
// describe.
//
// Compiled into ddx_rt only when DDX_NO_EXCEPTIONS is on; with exceptions
// enabled Boost defines these itself and this file is not built.
#include <boost/assert/source_location.hpp>
#include <boost/throw_exception.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace boost {

void throw_exception(const std::exception &e) {
  std::fprintf(stderr, "ddx: boost threw with exceptions disabled: %s\n",
               e.what());
  std::abort();
}

void throw_exception(const std::exception &e, const source_location &loc) {
  std::fprintf(stderr,
               "ddx: boost threw with exceptions disabled: %s (%s:%u)\n",
               e.what(), loc.file_name(), loc.line());
  std::abort();
}

} // namespace boost
