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

if (DDX_BUILD_RT)
    target_link_libraries(consumer PRIVATE ddx::rt)
    target_compile_definitions(consumer PRIVATE DDX_CONSUMER_RT)
endif ()
if (DDX_BUILD_JIT)
    # DDX_HAS_JIT arrives with the target, so consumer.cpp needs no flag of ours.
    target_link_libraries(consumer PRIVATE ddx::jit)
endif ()

enable_testing()
add_test(NAME consumer_runs COMMAND consumer)
