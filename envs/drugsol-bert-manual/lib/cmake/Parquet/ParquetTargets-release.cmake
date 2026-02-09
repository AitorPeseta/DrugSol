#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Parquet::parquet_shared" for configuration "Release"
set_property(TARGET Parquet::parquet_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Parquet::parquet_shared PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "thrift::thrift;opentelemetry-cpp::trace;opentelemetry-cpp::logs;opentelemetry-cpp::otlp_http_log_record_exporter;opentelemetry-cpp::ostream_log_record_exporter;opentelemetry-cpp::ostream_span_exporter;opentelemetry-cpp::otlp_http_exporter"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libparquet.so.2300.0.0"
  IMPORTED_SONAME_RELEASE "libparquet.so.2300"
  )

list(APPEND _cmake_import_check_targets Parquet::parquet_shared )
list(APPEND _cmake_import_check_files_for_Parquet::parquet_shared "${_IMPORT_PREFIX}/lib/libparquet.so.2300.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
