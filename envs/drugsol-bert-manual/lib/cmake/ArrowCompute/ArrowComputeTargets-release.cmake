#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ArrowCompute::arrow_compute_shared" for configuration "Release"
set_property(TARGET ArrowCompute::arrow_compute_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ArrowCompute::arrow_compute_shared PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "opentelemetry-cpp::trace;opentelemetry-cpp::logs;opentelemetry-cpp::otlp_http_log_record_exporter;opentelemetry-cpp::ostream_log_record_exporter;opentelemetry-cpp::ostream_span_exporter;opentelemetry-cpp::otlp_http_exporter;re2::re2"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libarrow_compute.so.2300.0.0"
  IMPORTED_SONAME_RELEASE "libarrow_compute.so.2300"
  )

list(APPEND _cmake_import_check_targets ArrowCompute::arrow_compute_shared )
list(APPEND _cmake_import_check_files_for_ArrowCompute::arrow_compute_shared "${_IMPORT_PREFIX}/lib/libarrow_compute.so.2300.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
