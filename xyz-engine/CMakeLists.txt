# xyz-engine: the speculative decode rounds of the Qwen3.8-27B PTQ1_0 hybrid + the xyz v1.2 drafter as one CUDA DLL, loaded
# by llama-server when XYZ_ENGINE=1. Added from the top-level CMakeLists.txt after ggml (Windows + GGML_CUDA only: the accept
# step reproduces MSVC's expf, see expf_exc.bin).
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/EHsc /Zc:preprocessor\"")
add_compile_definitions(_WINDOWS NDEBUG GGML_BACKEND_BUILD GGML_BACKEND_SHARED _CRT_SECURE_NO_WARNINGS GGML_SCHED_MAX_COPIES=4
                        _XOPEN_SOURCE=600 GGML_CUDA_PEER_MAX_BATCH_SIZE=128 GGML_CUDA_USE_GRAPHS GGML_SHARED NOMINMAX)

# The accept step matches MSVC IEEE float/double arithmetic: fast math would flush denormals and replace expf/division,
# and FMA contraction would change the reduction result.
add_library(xe_exact STATIC src/k_accept.cu)
target_include_directories(xe_exact PRIVATE src)
target_compile_options(xe_exact PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)

# ggml-cuda's own nvcc options: the engine's kernels are bit-identical to the fork's at these flags
set(ENGINE_SRC src/api.cu src/model.cu src/engine.cu src/fork_shims.cu src/draft.cu src/session.cu src/prefill.cu
    src/k_hadamard.cu src/k_attn_chain.cu src/k_small.cu src/k_ptq1_own.cu src/k_gdn_own.cu src/k_gdn_pro.cu src/k_q3k.cu
    src/k_draw.cu src/k_fa.cu src/k_mmq.cu)
add_library(xyz_engine SHARED ${ENGINE_SRC})
target_compile_options(xyz_engine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -extended-lambda -compress-mode=size>)
target_include_directories(xyz_engine PRIVATE include src ${PROJECT_SOURCE_DIR}/ggml/include ${PROJECT_SOURCE_DIR}/ggml/src
                           ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
target_link_libraries(xyz_engine PRIVATE xe_exact ggml-base CUDA::cudart CUDA::cublas CUDA::cuda_driver)
# the server loads the table from the DLL's directory
add_custom_command(TARGET xyz_engine POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/expf_exc.bin $<TARGET_FILE_DIR:xyz_engine>)

# regenerates expf_exc.bin (no fast math): expf_table <out>
add_executable(expf_table EXCLUDE_FROM_ALL tools/expf_table.cu)
target_link_libraries(expf_table PRIVATE CUDA::cudart)
