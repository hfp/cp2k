#!-------------------------------------------------------------------------------------------------!
#!   CP2K: A general program to perform molecular dynamics simulations                             !
#!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                  !
#!                                                                                                 !
#!   SPDX-License-Identifier: GPL-2.0-or-later                                                     !
#!-------------------------------------------------------------------------------------------------!

include(FindPackageHandleStandardArgs)
include(cp2k_utils)
find_package(PkgConfig QUIET)

cp2k_set_default_paths(LIBXS "LibXS")

if(PKG_CONFIG_FOUND)
  if(BUILD_SHARED_LIBS)
    pkg_check_modules(CP2K_LIBXS IMPORTED_TARGET GLOBAL libxs-shared)
  else()
    pkg_check_modules(CP2K_LIBXS IMPORTED_TARGET GLOBAL libxs-static)
  endif()
  if(NOT CP2K_LIBXS_FOUND)
    pkg_check_modules(CP2K_LIBXS IMPORTED_TARGET GLOBAL libxs)
  endif()
endif()

if(NOT CP2K_LIBXS_FOUND)
  cp2k_find_libraries(LIBXS xs)
endif()

if(NOT CP2K_LIBXS_INCLUDE_DIRS)
  cp2k_include_dirs(LIBXS "libxs.h;include/libxs.h")
endif()

if(CP2K_LIBXS_INCLUDE_DIRS)
  find_package_handle_standard_args(LibXS DEFAULT_MSG CP2K_LIBXS_INCLUDE_DIRS
                                    CP2K_LIBXS_LINK_LIBRARIES)
else()
  find_package_handle_standard_args(LibXS DEFAULT_MSG CP2K_LIBXS_LINK_LIBRARIES)
endif()

if(NOT TARGET cp2k::LibXS::libxs)
  add_library(cp2k::LibXS::libxs INTERFACE IMPORTED)
  if(CP2K_LIBXS_FOUND)
    if(CP2K_LIBXS_LIBRARY_DIRS)
      target_link_directories(cp2k::LibXS::libxs INTERFACE
                              ${CP2K_LIBXS_LIBRARY_DIRS})
    endif()
    set_target_properties(
      cp2k::LibXS::libxs PROPERTIES INTERFACE_LINK_LIBRARIES
                                    "${CP2K_LIBXS_LINK_LIBRARIES}")
    if(CP2K_LIBXS_INCLUDE_DIRS)
      set_target_properties(
        cp2k::LibXS::libxs
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                   "${CP2K_LIBXS_INCLUDE_DIRS};${CP2K_LIBXS_PREFIX}/include")
    endif()
  endif()
endif()

if(NOT TARGET cp2k::LibXS)
  add_library(cp2k::LibXS INTERFACE IMPORTED)
  target_link_libraries(cp2k::LibXS INTERFACE cp2k::LibXS::libxs)
endif()

mark_as_advanced(CP2K_LIBXS_INCLUDE_DIRS CP2K_LIBXS_LIBRARY_DIRS
                 CP2K_LIBXS_LINK_LIBRARIES)
