cmake_minimum_required(VERSION 3.20)

project(analysis_helpers VERSION 0.1.0 LANGUAGES CXX)

# Base C++ toolchain defaults for future native modules/classes.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ANALYSIS_HELPERS_BUILD_CPP "Build C++ targets" ON)
option(ANALYSIS_HELPERS_BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(ANALYSIS_HELPERS_BUILD_ROOT_CLASSES "Build ROOT-based C++ classes" ON)

set(ANALYSIS_HELPERS_HAS_ROOT OFF)

# Keep the layout explicit so we can incrementally add targets.
set(AH_CPP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")
set(AH_CPP_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp/src")
set(AH_CPP_BINDINGS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp/bindings")

if(ANALYSIS_HELPERS_BUILD_CPP)
  message(STATUS "C++ scaffolding enabled: add library targets under cpp/src")
endif()

if(ANALYSIS_HELPERS_BUILD_ROOT_CLASSES)
  find_package(ROOT QUIET COMPONENTS Core MathCore RooFitCore RooFit)

  if(ROOT_FOUND)
    set(ANALYSIS_HELPERS_HAS_ROOT ON)
    include(${ROOT_USE_FILE})

    set(AH_ROOT_DICT_HEADERS
      ${AH_CPP_INCLUDE_DIR}/RooThreshold.h
      ${AH_CPP_INCLUDE_DIR}/RooThresholdDAcp.h
    )

    set(AH_ROOT_LINKDEF ${AH_CPP_INCLUDE_DIR}/LinkDef.h)

    ROOT_GENERATE_DICTIONARY(
      G__analysis_helpers_root
      ${AH_ROOT_DICT_HEADERS}
      LINKDEF ${AH_ROOT_LINKDEF}
    )

    add_library(analysis_helpers_root_classes SHARED
      ${AH_CPP_SRC_DIR}/RooThreshold.cpp
      ${AH_CPP_SRC_DIR}/RooThresholdDAcp.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/G__analysis_helpers_root.cxx
    )

    target_include_directories(analysis_helpers_root_classes
      PUBLIC
        ${AH_CPP_INCLUDE_DIR}
    )

    target_link_libraries(analysis_helpers_root_classes
      PUBLIC
        ROOT::Core
        ROOT::MathCore
        ROOT::RooFitCore
        ROOT::RooFit
    )

    enable_testing()

    add_executable(test_roo_threshold
      ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/test_roo_threshold.cpp
    )

    target_include_directories(test_roo_threshold
      PRIVATE
        ${AH_CPP_INCLUDE_DIR}
    )

    target_link_libraries(test_roo_threshold
      PRIVATE
        analysis_helpers_root_classes
        ROOT::Core
        ROOT::MathCore
        ROOT::RooFitCore
        ROOT::RooFit
    )

    add_test(NAME roo_threshold_integration COMMAND test_roo_threshold)
  else()
    message(WARNING "ROOT not found: skipping ROOT-based classes and ROOT C++ tests")
  endif()
endif()

if(ANALYSIS_HELPERS_BUILD_PYTHON_BINDINGS)
  if(NOT ANALYSIS_HELPERS_HAS_ROOT)
    message(WARNING "Python bindings requested, but ROOT classes are unavailable; skipping _core module")
  else()
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_core MODULE
      ${AH_CPP_BINDINGS_DIR}/module.cpp
    )

    target_include_directories(_core
      PRIVATE
        ${AH_CPP_INCLUDE_DIR}
    )

    target_link_libraries(_core
      PRIVATE
        analysis_helpers_root_classes
    )

    install(TARGETS _core LIBRARY DESTINATION analysis_helpers/_cpp)

    add_test(
      NAME python_cpp_binding_smoke
      COMMAND
        ${Python_EXECUTABLE} -c
        "import sys; sys.path.insert(0, r'$<TARGET_FILE_DIR:_core>'); import _core; v1 = _core.evaluate_threshold(2.0, 1.0, 1.5); v2 = _core.evaluate_threshold_dacp(2.0, 1.0, 1.5, 2.0); assert v1 > 0 and v2 > 0"
    )
  endif()
endif()
