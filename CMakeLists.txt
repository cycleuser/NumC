cmake_minimum_required(VERSION 3.10)
project(NumC C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(NC_BUILD_TESTS "Build tests" ON)
option(NC_BUILD_EXAMPLES "Build examples" ON)
option(NC_USE_OPENMP "Enable OpenMP support" OFF)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

set(NC_SOURCES
    src/NumC.c
)

add_library(numc STATIC ${NC_SOURCES})

set_target_properties(numc PROPERTIES
    VERSION 0.1.0
    SOVERSION 1
)

target_include_directories(numc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(NC_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(numc PUBLIC OpenMP::OpenMP_C)
endif()

if(NC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(NC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

install(TARGETS numc
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
