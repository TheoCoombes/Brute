cmake_minimum_required(VERSION 3.18...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _torch_rc
)
if(NOT _torch_rc EQUAL 0)
  message(FATAL_ERROR "PyTorch not found. Run: pip install torch")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Optional CUDA
if(CMAKE_CUDA_COMPILER OR Torch_CUDA)
  enable_language(CUDA)
  set(HAVE_CUDA TRUE)
  message(STATUS "brute: CUDA backend enabled")
  # Build for Volta → Hopper. CUTLASS B1 XOR-popc tensor-core path activates on sm_80+;
  # earlier sms fall back to the hand-tuned XNOR kernel automatically.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
  endif()
endif()

# Metal/MPS (macOS only)
if(APPLE)
  enable_language(OBJCXX)
  set(HAVE_MPS TRUE)
  message(STATUS "brute: Metal/MPS backend enabled")
endif()

# ── Google Highway (vendored submodule, static dispatch) ──────────────────────
set(HWY_ENABLE_CONTRIB  OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS    OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL  OFF CACHE BOOL "" FORCE)
set(HWY_FORCE_STATIC_LIBS ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/highway EXCLUDE_FROM_ALL)

# Sources
set(BRUTE_SOURCES
  cbrute/ext.cpp
  cbrute/cpu/ops_cpu.cpp
)
if(HAVE_CUDA)
  list(APPEND BRUTE_SOURCES cbrute/cuda/ops_cuda.cu)
endif()
if(HAVE_MPS)
  list(APPEND BRUTE_SOURCES cbrute/metal/ops_metal.mm)
  set_source_files_properties(cbrute/metal/ops_metal.mm
    PROPERTIES COMPILE_FLAGS "-x objective-c++")

  # ── AOT MSL kernel compilation ──────────────────────────────────────────────
  # Each .metal source → .air via `xcrun metal`, all linked into a single
  # .metallib via `xcrun metallib`, then embedded into a generated header
  # (brute_kernels_data.h) and #included by ops_metal.mm. The result is one
  # .so with kernels validated at build time and zero runtime compile cost.
  find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
  set(BRUTE_METAL_SRCS
    cbrute/metal/kernels/packing.metal
    cbrute/metal/kernels/popcount.metal
    cbrute/metal/kernels/reductions.metal
    cbrute/metal/kernels/matmul.metal
    cbrute/metal/kernels/bitwise.metal
  )
  set(BRUTE_METAL_AIRS "")
  foreach(src IN LISTS BRUTE_METAL_SRCS)
    get_filename_component(name "${src}" NAME_WE)
    set(air "${CMAKE_CURRENT_BINARY_DIR}/${name}.air")
    add_custom_command(
      OUTPUT  "${air}"
      COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
              -c -O3 -ffast-math
              -I "${CMAKE_CURRENT_SOURCE_DIR}/cbrute/metal/kernels"
              "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
              -o "${air}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
      COMMENT "metal: compiling ${name}.air"
      VERBATIM
    )
    list(APPEND BRUTE_METAL_AIRS "${air}")
  endforeach()
  set(BRUTE_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/brute_kernels.metallib")
  add_custom_command(
    OUTPUT  "${BRUTE_METALLIB}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib
            ${BRUTE_METAL_AIRS} -o "${BRUTE_METALLIB}"
    DEPENDS ${BRUTE_METAL_AIRS}
    COMMENT "metal: linking brute_kernels.metallib"
    VERBATIM
  )
  set(BRUTE_METALLIB_H "${CMAKE_CURRENT_BINARY_DIR}/brute_kernels_data.h")
  add_custom_command(
    OUTPUT  "${BRUTE_METALLIB_H}"
    COMMAND "${CMAKE_COMMAND}"
            -DIN=${BRUTE_METALLIB}
            -DOUT=${BRUTE_METALLIB_H}
            -DSYM=kBruteKernelsMetallib
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
    DEPENDS "${BRUTE_METALLIB}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
    COMMENT "metal: embedding metallib bytes into brute_kernels_data.h"
    VERBATIM
  )
  add_custom_target(brute_metallib DEPENDS "${BRUTE_METALLIB_H}")
endif()

python_add_library(_cbrute MODULE ${BRUTE_SOURCES} WITH_SOABI)
if(HAVE_MPS)
  # Ensure the generated metallib header exists before ops_metal.mm compiles.
  add_dependencies(_cbrute brute_metallib)
endif()

target_include_directories(_cbrute PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/cbrute"
  "${CMAKE_CURRENT_SOURCE_DIR}/cbrute/cpu"
  "${CMAKE_CURRENT_SOURCE_DIR}/cbrute/metal"
  "${CMAKE_CURRENT_SOURCE_DIR}/cbrute/cuda"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/highway"
  "${CMAKE_CURRENT_BINARY_DIR}"            # for brute_kernels_data.h (HAVE_MPS)
  ${TORCH_INCLUDE_DIRS}
)

if(HAVE_CUDA)
  # CUTLASS is header-only; we only need its include tree (no add_subdirectory
  # — that would build their tests and tools).
  target_include_directories(_cbrute PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include"
  )
endif()

target_link_libraries(_cbrute PRIVATE ${TORCH_LIBRARIES} hwy)

if(HAVE_MPS)
  target_link_libraries(_cbrute PRIVATE
    "-framework Metal"
    "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph"
    "-framework Foundation")
endif()

target_compile_definitions(_cbrute PRIVATE
  $<$<BOOL:${HAVE_CUDA}>:HAVE_CUDA>
  $<$<BOOL:${HAVE_MPS}>:HAVE_MPS>
)

# Optimize for the host CPU and enable Highway's recommended baseline.
# (Torch typically already sets -O3; we add -march=native for SIMD breadth.)
if(NOT MSVC)
  target_compile_options(_cbrute PRIVATE -O3 -fno-math-errno -fno-trapping-math)
  if(NOT APPLE OR CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
    target_compile_options(_cbrute PRIVATE -march=native)
  endif()
endif()

set_property(TARGET _cbrute PROPERTY CXX_STANDARD 17)
if(HAVE_CUDA)
  set_property(TARGET _cbrute PROPERTY CUDA_STANDARD 17)
endif()

install(TARGETS _cbrute LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
