#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "prometheus-cpp::core" for configuration "Release"
set_property(TARGET prometheus-cpp::core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(prometheus-cpp::core PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libprometheus-cpp-core.so.1.3.0"
  IMPORTED_SONAME_RELEASE "libprometheus-cpp-core.so.1.3"
  )

list(APPEND _cmake_import_check_targets prometheus-cpp::core )
list(APPEND _cmake_import_check_files_for_prometheus-cpp::core "${_IMPORT_PREFIX}/lib/libprometheus-cpp-core.so.1.3.0" )

# Import target "prometheus-cpp::pull" for configuration "Release"
set_property(TARGET prometheus-cpp::pull APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(prometheus-cpp::pull PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libprometheus-cpp-pull.so.1.3.0"
  IMPORTED_SONAME_RELEASE "libprometheus-cpp-pull.so.1.3"
  )

list(APPEND _cmake_import_check_targets prometheus-cpp::pull )
list(APPEND _cmake_import_check_files_for_prometheus-cpp::pull "${_IMPORT_PREFIX}/lib/libprometheus-cpp-pull.so.1.3.0" )

# Import target "prometheus-cpp::push" for configuration "Release"
set_property(TARGET prometheus-cpp::push APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(prometheus-cpp::push PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libprometheus-cpp-push.so.1.3.0"
  IMPORTED_SONAME_RELEASE "libprometheus-cpp-push.so.1.3"
  )

list(APPEND _cmake_import_check_targets prometheus-cpp::push )
list(APPEND _cmake_import_check_files_for_prometheus-cpp::push "${_IMPORT_PREFIX}/lib/libprometheus-cpp-push.so.1.3.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
