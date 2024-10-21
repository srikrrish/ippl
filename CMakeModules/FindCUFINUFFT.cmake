#
# Find CUFINUFFT includes and library
#
# CUFINUFFT_INCLUDE_DIR - where to find cufinufft.h
# CUFINUFFT_LIBRARY     - libcufinufft.so path
# CUFINUFFT_FOUND       - do not attempt to use if "no" or undefined.

FIND_PATH(CUFINUFFT_INCLUDE_DIR cufinufft.h
    HINTS $ENV{CUFINUFFT_INCLUDE_PATH} $ENV{CUFINUFFT_INCLUDE_DIR} $ENV{CUFINUFFT_PREFIX}/include $ENV{CUFINUFFT_DIR}/include ${PROJECT_SOURCE_DIR}/include
    PATHS ENV CPP_INCLUDE_PATH
)
#Static library has some issues and gives a cuda error at the end of compilation
FIND_LIBRARY(CUFINUFFT_LIBRARY_DIR libcufinufft.a
    HINTS $ENV{CUFINUFFT_LIBRARY_PATH} $ENV{CUFINUFFT_LIBRARY_DIR} $ENV{CUFINUFFT_PREFIX}/lib $ENV{CUFINUFFT_DIR}/lib $ENV{CUFINUFFT}/lib ${PROJECT_SOURCE_DIR}/lib
    PATHS ENV LIBRARY_PATH
)

IF(CUFINUFFT_INCLUDE_DIR AND CUFINUFFT_LIBRARY_DIR)
    SET( CUFINUFFT_FOUND "YES" )
    SET( CUFINUFFT_DIR $ENV{CUFINUFFT_DIR} )
ENDIF()

IF (CUFINUFFT_FOUND)
   IF (NOT CUFINUFFT_FIND_QUIETLY)
       MESSAGE(STATUS "Found cufinufft library dir: ${CUFINUFFT_LIBRARY_DIR}")
       MESSAGE(STATUS "Found cufinufft include dir: ${CUFINUFFT_INCLUDE_DIR}")
   ENDIF (NOT CUFINUFFT_FIND_QUIETLY)
ELSE (CUFINUFFT_FOUND)
    IF (CUFINUFFT_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find CUFINUFFT!")
  ENDIF (CUFINUFFT_FIND_REQUIRED)
ENDIF (CUFINUFFT_FOUND)