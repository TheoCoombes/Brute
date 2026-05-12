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
endif()

# Metal/MPS (macOS only)
if(APPLE)
  enable_language(OBJCXX)
  set(HAVE_MPS TRUE)
  message(STATUS "brute: Metal/MPS backend enabled")
endif()

# libpopcnt – header-only
add_library(libpopcnt INTERFACE)
target_include_directories(libpopcnt INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libpopcnt")

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
  set_source_files_properties(cbrute/backend/metal/ops_metal.mm
    PROPERTIES COMPILE_FLAGS "-x objective-c++")
endif()

python_add_library(_cbrute MODULE ${BRUTE_SOURCES} WITH_SOABI)

target_include_directories(_cbrute PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/cbrute/cpu"
  "${CMAKE_CURRENT_SOURCE_DIR}/cbrute/metal"
  "${CMAKE_CURRENT_SOURCE_DIR}/cbrute/cuda"
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(_cbrute PRIVATE ${TORCH_LIBRARIES} libpopcnt)

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

set_property(TARGET _cbrute PROPERTY CXX_STANDARD 17)
if(HAVE_CUDA)
  set_property(TARGET _cbrute PROPERTY CUDA_STANDARD 17)
endif()

install(TARGETS _cbrute LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})