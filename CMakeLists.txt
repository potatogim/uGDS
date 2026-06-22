cmake_minimum_required(VERSION 3.18)
project(ugds LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dependencies
find_package(Threads REQUIRED)

# - GPU Backend Selection ---------------------
option(UGDS_BACKEND_CUDA "Use NVIDIA CUDA backend" ON)
option(UGDS_BACKEND_HIP  "Use AMD HIP backend"     OFF)

if(UGDS_BACKEND_CUDA AND UGDS_BACKEND_HIP)
    message(FATAL_ERROR "Cannot enable both CUDA and HIP backends simultaneously")
endif()

# HIP language support requires CMake >= 3.21
if(UGDS_BACKEND_HIP AND CMAKE_VERSION VERSION_LESS "3.21")
    message(FATAL_ERROR "HIP backend requires CMake >= 3.21 (found ${CMAKE_VERSION})")
endif()

if(UGDS_BACKEND_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(UGDS_GPU_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    set(UGDS_GPU_LIBS "")
    set(UGDS_GPU_LANGUAGE CUDA)
    set(UGDS_GPU_ARCHITECTURES "80;90")
elseif(UGDS_BACKEND_HIP)
    # Set HIP architectures before enable_language to avoid CMake auto-probing
    if(NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx942" CACHE STRING "HIP target architectures")  # MI250, MI300X
    endif()
    find_package(hip REQUIRED)
    # Respect CMAKE_PREFIX_PATH for ROCm install location; fallback to /opt/rocm
    if(NOT DEFINED ROCM_PATH)
        find_path(ROCM_PATH include/hsa/hsa_ext_amd.h PATHS /opt/rocm ${CMAKE_PREFIX_PATH})
    endif()
    set(UGDS_GPU_INCLUDE_DIRS ${HIP_INCLUDE_DIRS})
    if(ROCM_PATH)
        list(APPEND UGDS_GPU_INCLUDE_DIRS ${ROCM_PATH}/include)
    endif()
    set(UGDS_GPU_LIBS hip::host)
    set(UGDS_GPU_LANGUAGE HIP)
    set(UGDS_GPU_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

    # HSA runtime for hsa_amd_portable_export_dmabuf / close_dmabuf
    find_package(hsa-runtime64 CONFIG QUIET)
    if(hsa-runtime64_FOUND)
        list(APPEND UGDS_GPU_LIBS hsa-runtime64::hsa-runtime64)
    else()
        find_library(HSA_RUNTIME_LIB hsa-runtime64 PATHS ${ROCM_PATH}/lib /opt/rocm/lib REQUIRED)
        list(APPEND UGDS_GPU_LIBS ${HSA_RUNTIME_LIB})
    endif()

    # Detect hsa_amd_portable_export_dmabuf_v2 (ROCm >= 5.6 with PCIe P2P mapping)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES "${UGDS_GPU_INCLUDE_DIRS}")
    set(CMAKE_REQUIRED_LIBRARIES "${UGDS_GPU_LIBS}")
    check_cxx_source_compiles("
        #include <hsa/hsa_ext_amd.h>
        int main() {
            int fd; uint64_t off;
            hsa_amd_portable_export_dmabuf_v2((const void*)0, 0, &fd, &off,
                                               HSA_AMD_DMABUF_MAPPING_TYPE_PCIE);
            return 0;
        }
    " HAVE_HSA_DMABUF_V2)
    set(CMAKE_REQUIRED_INCLUDES)
    set(CMAKE_REQUIRED_LIBRARIES)
    if(NOT HAVE_HSA_DMABUF_V2)
        message(FATAL_ERROR "HIP backend requires hsa_amd_portable_export_dmabuf_v2 "
                            "(HSA_AMD_DMABUF_MAPPING_TYPE_PCIE). "
                            "Please install ROCm >= 5.6 or disable HIP backend (-DUGDS_BACKEND_HIP=OFF).")
    endif()
else()
    message(WARNING "No GPU backend selected, building library only")
    set(UGDS_GPU_LANGUAGE "")
endif()

# Library sources
set(UGDS_SOURCES
    src/libnvm/admin.cpp
    src/libnvm/ctrl.cpp
    src/libnvm/dma.cpp
    src/libnvm/error.cpp
    src/libnvm/linux_device.cpp
    src/libnvm/linux_dma.cpp
    src/libnvm/mutex.cpp
    src/libnvm/queue.cpp
    src/libnvm/rpc.cpp
    src/ugds_driver.cpp
    src/ugds_handle.cpp
    src/ugds_buf.cpp
    src/ugds_io.cpp
)

add_library(ugds SHARED ${UGDS_SOURCES})

target_include_directories(ugds
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/libnvm
)

if(UGDS_BACKEND_CUDA)
    target_include_directories(ugds PRIVATE ${UGDS_GPU_INCLUDE_DIRS})
    target_compile_definitions(ugds PRIVATE _CUDA)
elseif(UGDS_BACKEND_HIP)
    target_include_directories(ugds PRIVATE ${UGDS_GPU_INCLUDE_DIRS})
    target_link_libraries(ugds PRIVATE ${UGDS_GPU_LIBS})
    target_compile_definitions(ugds PRIVATE _HIP)
    if(HAVE_HSA_DMABUF_V2)
        target_compile_definitions(ugds PRIVATE HAVE_HSA_DMABUF_V2)
    endif()
endif()

target_link_libraries(ugds PRIVATE Threads::Threads)
target_compile_options(ugds PRIVATE -Wall -O2)

# - Tests and benchmarks (need GPU for cudaMalloc/hipMalloc) ----
if(UGDS_GPU_LANGUAGE)
    include(CheckLanguage)
    check_language(${UGDS_GPU_LANGUAGE})
    if(CMAKE_${UGDS_GPU_LANGUAGE}_COMPILER)
        enable_language(${UGDS_GPU_LANGUAGE})
        set(CMAKE_${UGDS_GPU_LANGUAGE}_STANDARD 17)

        # - Functional regression tests (one executable per test case) -
        set(FUNCTIONAL_TESTS
            test_driver_lifecycle
            test_handle_register
            test_buf_register
            test_read_write_basic
            test_read_write_large
            test_read_write_unregistered
            test_alignment_errors
            test_multi_offset
            test_concurrent_qps
        )

        foreach(test_name ${FUNCTIONAL_TESTS})
            set(test_src ${CMAKE_CURRENT_SOURCE_DIR}/test/functional/${test_name}.cu)

            if(UGDS_BACKEND_HIP)
                # Compile .cu files as HIP language
                set_source_files_properties(${test_src} PROPERTIES LANGUAGE HIP)
            endif()

            add_executable(${test_name} ${test_src})
            target_include_directories(${test_name} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/test/functional
            )

            if(UGDS_BACKEND_CUDA)
                target_link_libraries(${test_name} ugds CUDA::cudart)
            elseif(UGDS_BACKEND_HIP)
                target_link_libraries(${test_name} ugds hip::host ${HSA_RUNTIME_LIB})
                target_compile_definitions(${test_name} PRIVATE __HIP_PLATFORM_AMD__)
            endif()

            set_target_properties(${test_name} PROPERTIES
                ${UGDS_GPU_LANGUAGE}_ARCHITECTURES "${UGDS_GPU_ARCHITECTURES}"
                BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
            )
        endforeach()

        # - Performance benchmarks -

        set(bench_src ${CMAKE_CURRENT_SOURCE_DIR}/test/perf/cufile_bench.cu)
        if(UGDS_BACKEND_HIP)
            set_source_files_properties(${bench_src} PROPERTIES LANGUAGE HIP)
        endif()

        # bench_ugds: uGDS benchmark linked against libugds.so
        add_executable(bench_ugds ${bench_src})
        target_include_directories(bench_ugds PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/test/perf
        )
        if(UGDS_BACKEND_CUDA)
            target_link_libraries(bench_ugds ugds CUDA::cudart)
        elseif(UGDS_BACKEND_HIP)
            target_link_libraries(bench_ugds ugds hip::host ${HSA_RUNTIME_LIB})
            target_compile_definitions(bench_ugds PRIVATE __HIP_PLATFORM_AMD__)
        endif()
        set_target_properties(bench_ugds PROPERTIES
            ${UGDS_GPU_LANGUAGE}_ARCHITECTURES "${UGDS_GPU_ARCHITECTURES}"
            BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
        )

        # bench_gds: same benchmark linked against NVIDIA's libcufile.so (optional)
        set(CUFILE_LIB "" CACHE FILEPATH "Path to NVIDIA libcufile.so for GDS comparison benchmark")
        set(CUFILE_INCLUDE "" CACHE PATH "Path to NVIDIA cuFile headers (if different from ours)")
        if(CUFILE_LIB)
            if(NOT UGDS_BACKEND_CUDA)
                message(WARNING "CUFILE_LIB is set but CUDA backend is not enabled. bench_gds requires CUDA; skipping.")
            else()
            add_executable(bench_gds ${bench_src})
            target_include_directories(bench_gds PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/test/perf
            )
            if(CUFILE_INCLUDE)
                target_include_directories(bench_gds PRIVATE ${CUFILE_INCLUDE})
            else()
                target_include_directories(bench_gds PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
            endif()
            target_link_libraries(bench_gds ${CUFILE_LIB} CUDA::cudart)
            target_compile_definitions(bench_gds PRIVATE USE_O_DIRECT USE_NVIDIA_GDS)
            set_target_properties(bench_gds PROPERTIES
                CUDA_ARCHITECTURES "80;90"
            )
            endif()
        endif()

    else()
        message(WARNING "${UGDS_GPU_LANGUAGE} compiler not found, tests will not be built")
    endif()
else()
    message(WARNING "No GPU backend enabled, tests will not be built")
endif()
