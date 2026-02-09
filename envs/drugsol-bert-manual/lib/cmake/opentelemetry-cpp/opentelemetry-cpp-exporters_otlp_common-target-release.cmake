#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "opentelemetry-cpp::otlp_recordable" for configuration "Release"
set_property(TARGET opentelemetry-cpp::otlp_recordable APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::otlp_recordable PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_otlp_recordable.so"
  IMPORTED_SONAME_RELEASE "libopentelemetry_otlp_recordable.so"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::otlp_recordable )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::otlp_recordable "${_IMPORT_PREFIX}/lib/libopentelemetry_otlp_recordable.so" )

# Import target "opentelemetry-cpp::proto" for configuration "Release"
set_property(TARGET opentelemetry-cpp::proto APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::proto PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_proto.so"
  IMPORTED_SONAME_RELEASE "libopentelemetry_proto.so"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::proto )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::proto "${_IMPORT_PREFIX}/lib/libopentelemetry_proto.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
