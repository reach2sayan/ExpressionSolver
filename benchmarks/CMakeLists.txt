ddx_use_googlebenchmark()

add_executable(benchmarks benchmark.cpp)
target_link_libraries(benchmarks PRIVATE benchmark::benchmark ddx)
ddx_target_flags(benchmarks)

if (DDX_BUILD_JIT)
    add_executable(benchmarks_jit jit/benchmark_jit.cpp)
    target_link_libraries(benchmarks_jit PRIVATE benchmark::benchmark ddx::jit)
    ddx_target_flags(benchmarks_jit)
    ddx_runtime_deps(benchmarks_jit)

    # Not a Google Benchmark loop: it compiles one graph per cell and reports
    # where the time went, so it links ddx::jit alone.
    add_executable(benchmarks_jit_compile jit/benchmark_jit_compile.cpp)
    target_link_libraries(benchmarks_jit_compile PRIVATE ddx::jit)
    ddx_target_flags(benchmarks_jit_compile)
    ddx_runtime_deps(benchmarks_jit_compile)
endif ()

add_custom_target(benchmark_json
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/benchmark-results"
        COMMAND $<TARGET_FILE:benchmarks>
        --benchmark_min_time=0.05s
        --benchmark_out="${CMAKE_BINARY_DIR}/benchmark-results/benchmarks.json"
        --benchmark_out_format=json
        DEPENDS benchmarks
        USES_TERMINAL
        COMMENT "Running benchmarks and exporting JSON results")
