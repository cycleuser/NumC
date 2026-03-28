include_directories(${CMAKE_SOURCE_DIR}/../include)

add_executable(test_numc
    test_numc.c
)

target_link_libraries(test_numc numc)

add_test(NAME test_numc COMMAND test_numc)
