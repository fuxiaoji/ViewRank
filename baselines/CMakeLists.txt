cmake_minimum_required(VERSION 3.20)
project(MaturePRPACK LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(PRPACK_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/prpack)
set(PRPACK_FILES prpack_utils.cpp prpack_base_graph.cpp
  prpack_preprocessed_ge_graph.cpp prpack_preprocessed_gs_graph.cpp
  prpack_preprocessed_schur_graph.cpp prpack_preprocessed_scc_graph.cpp
  prpack_solver.cpp prpack_solver_ge.cpp prpack_solver_sccgs.cpp
  prpack_solver_inout.cpp prpack_result.cpp)
list(TRANSFORM PRPACK_FILES PREPEND "${PRPACK_SOURCE}/")
add_library(prpack_native SHARED ${PRPACK_FILES}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/baselines/prpack_adapter.cpp)
target_include_directories(prpack_native PRIVATE ${PRPACK_SOURCE})
target_compile_definitions(prpack_native PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
if(MSVC)
  target_compile_options(prpack_native PRIVATE /O2 /EHsc)
endif()
