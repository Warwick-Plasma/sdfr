cmake_minimum_required(VERSION 3.14)

set(COMMIT_INFO _commit_info.py)
set(COMMIT_INFO_FILE "${OUTDIR}/${COMMIT_INFO}")
set(COMMIT_INFO_TEMPLATE ${COMMIT_INFO}.in)

execute_process(
    COMMAND ${GIT_EXECUTABLE} show-ref
    OUTPUT_QUIET
    ERROR_QUIET
    RESULT_VARIABLE GOTGIT)

if(GOTGIT EQUAL 0)
    # In a git repo
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        OUTPUT_VARIABLE COMMIT_ID
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    execute_process(
        COMMAND ${GIT_EXECUTABLE} log --pretty=format:%cd -1 HEAD
        OUTPUT_VARIABLE COMMIT_DATE)
else()
    set(COMMIT_ID "unknown")
    set(COMMIT_DATE "unknown")
endif()

configure_file(${COMMIT_INFO_TEMPLATE} ${COMMIT_INFO_FILE} @ONLY)
