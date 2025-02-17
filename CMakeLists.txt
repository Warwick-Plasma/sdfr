cmake_minimum_required(VERSION 3.21)
project(sdfr)

add_subdirectory(src/C C EXCLUDE_FROM_ALL)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

find_package(Git REQUIRED)

set(COMMIT_INFO _commit_info.py)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${COMMIT_INFO}
    COMMAND ${CMAKE_COMMAND} -DGIT_EXECUTABLE=${GIT_EXECUTABLE} -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR} -P CommitInfo.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_custom_target(sdfr ALL DEPENDS ${COMMIT_INFO} sdfc_shared)

# Install the shared library
install(TARGETS sdfc_shared DESTINATION ${SKBUILD_PROJECT_NAME})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${COMMIT_INFO} DESTINATION ${SKBUILD_PROJECT_NAME})
