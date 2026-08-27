# A consumer of ddx, as a separate project: this is configured and built by the
# tests in ../CMakeLists.txt, once against a build tree and once against an
# install prefix, and is never part of the parent build.  Its whole job is to
# fail when find_package(ddx) does not give a consumer what it needs.
cmake_minimum_required(VERSION 3.26)
project(ddx_package_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# What this consumer claims to need.  The driving test passes the components the
# ddx it is pointed at was built with -- and, for the negative case, one it was
# not, which must fail here rather than at link time.
set(DDX_CONSUMER_COMPONENTS "" CACHE STRING "Components to request of ddx")

if (DDX_CONSUMER_COMPONENTS)
    find_package(ddx REQUIRED COMPONENTS ${DDX_CONSUMER_COMPONENTS})
else ()
    find_package(ddx REQUIRED)
endif ()

message(STATUS "ddx ${ddx_VERSION} from ${ddx_DIR}")
message(STATUS "ddx built with: ${ddx_CODEGEN_FLAGS} (FP contraction: ${ddx_FP_CONTRACT})")

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE ddx::ddx)

# The flags ddx recorded, applied rather than guessed: this is the consumer-side
# half of what ddx_CODEGEN_FLAGS exists for.
target_compile_options(consumer PRIVATE ${ddx_CODEGEN_FLAGS})

# Driven off what find_package actually reported, not off cache variables the
# parent never passes down -- which is what these two blocks used to read, so
# the consumer had never once linked the runtime graph or the backend.
target_link_libraries(consumer PRIVATE ddx::rt)
target_compile_definitions(consumer PRIVATE DDX_CONSUMER_RT)
if (ddx_jit_FOUND)
    # DDX_HAS_JIT arrives with the target, so consumer.cpp needs no flag of ours.
    target_link_libraries(consumer PRIVATE ddx::jit)
endif ()

# Every header the package ships, compiled against the Boost the package ships.
# Complete only where no other Boost sits on the default include path: a machine
# with /usr/include/boost fills any gap in the prefix silently, so this proves
# the shipped subset is whole on CI, where no Boost is installed, and not on a
# developer box that has a distro one.  scripts/boost_headers.py --check is the
# guard that does not depend on the environment.
# consumer.cpp reaches most of them through rt/equation.hpp, but not rt/text/,
# rt/dot.hpp or dual/hessian.hpp -- so a Boost library named by one of those,
# and missing from DdxBoostHeaders.cmake, would install short and go unnoticed.
# Globbed from the include directory ddx reported rather than from a list here,
# so this cannot fall behind what is installed.
get_target_property(DDX_INCLUDE_DIRS ddx::util INTERFACE_INCLUDE_DIRECTORIES)
list(GET DDX_INCLUDE_DIRS 0 DDX_INCLUDE_DIR)
file(GLOB_RECURSE DDX_HEADERS RELATIVE "${DDX_INCLUDE_DIR}" "${DDX_INCLUDE_DIR}/*.hpp")
# An install prefix keeps ddx's headers and the Boost it ships in the same
# include directory, so the glob finds both.  Boost is what this is compiled
# *against*, not part of what ddx ships.
list(FILTER DDX_HEADERS EXCLUDE REGEX "^boost/")
if (NOT DDX_HEADERS)
    message(FATAL_ERROR "no ddx headers under ${DDX_INCLUDE_DIR}")
endif ()
list(SORT DDX_HEADERS)
set(DDX_ALL_HEADERS_CPP "")
foreach (header IN LISTS DDX_HEADERS)
    string(APPEND DDX_ALL_HEADERS_CPP "#include \"${header}\"\n")
endforeach ()
string(APPEND DDX_ALL_HEADERS_CPP "int main() { return 0; }\n")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/all_headers.cpp"
        CONTENT "${DDX_ALL_HEADERS_CPP}")

add_executable(all_headers "${CMAKE_CURRENT_BINARY_DIR}/all_headers.cpp")
target_link_libraries(all_headers PRIVATE ddx::ddx ddx::rt)
target_compile_options(all_headers PRIVATE ${ddx_CODEGEN_FLAGS})
if (ddx_jit_FOUND)
    target_link_libraries(all_headers PRIVATE ddx::jit)
endif ()

enable_testing()
add_test(NAME consumer_runs COMMAND consumer)
add_test(NAME all_headers_runs COMMAND all_headers)
