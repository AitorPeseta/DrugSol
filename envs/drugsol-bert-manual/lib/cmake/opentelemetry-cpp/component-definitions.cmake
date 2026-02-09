# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# Configured from opentelemetry-cpp/cmake/component-definitions.cmake.in

# ----------------------------------------------------------------------
# opentelmetry-cpp Built COMPONENT list
# ----------------------------------------------------------------------
set(OTEL_BUILT_COMPONENTS_LIST api;sdk;ext_common;ext_http_curl;exporters_otlp_common;exporters_otlp_grpc;exporters_otlp_http;exporters_ostream;exporters_in_memory;exporters_prometheus;exporters_zipkin)

# ----------------------------------------------------------------------
# COMPONENT to TARGET lists
# ----------------------------------------------------------------------

# COMPONENT api
set(COMPONENT_api_TARGETS
    opentelemetry-cpp::api
)

# COMPONENT sdk
set(COMPONENT_sdk_TARGETS
    opentelemetry-cpp::sdk
    opentelemetry-cpp::common
    opentelemetry-cpp::resources
    opentelemetry-cpp::version
    opentelemetry-cpp::logs
    opentelemetry-cpp::trace
    opentelemetry-cpp::metrics
)

# COMPONENT ext_common
set(COMPONENT_ext_common_TARGETS
    opentelemetry-cpp::ext
)

# COMPONENT ext_http_curl
set(COMPONENT_ext_http_curl_TARGETS
    opentelemetry-cpp::http_client_curl
)

# COMPONENT exporters_otlp_common
set(COMPONENT_exporters_otlp_common_TARGETS
    opentelemetry-cpp::otlp_recordable
    opentelemetry-cpp::proto
)

# COMPONENT exporters_otlp_grpc
set(COMPONENT_exporters_otlp_grpc_TARGETS
    opentelemetry-cpp::otlp_grpc_client
    opentelemetry-cpp::proto_grpc
    opentelemetry-cpp::otlp_grpc_exporter
    opentelemetry-cpp::otlp_grpc_log_record_exporter
    opentelemetry-cpp::otlp_grpc_metrics_exporter
)

# COMPONENT exporters_otlp_http
set(COMPONENT_exporters_otlp_http_TARGETS
    opentelemetry-cpp::otlp_http_client
    opentelemetry-cpp::otlp_http_exporter
    opentelemetry-cpp::otlp_http_log_record_exporter
    opentelemetry-cpp::otlp_http_metric_exporter
)

# COMPONENT exporters_ostream
set(COMPONENT_exporters_ostream_TARGETS
    opentelemetry-cpp::ostream_span_exporter
    opentelemetry-cpp::ostream_metrics_exporter
    opentelemetry-cpp::ostream_log_record_exporter
)

# COMPONENT exporters_in_memory
set(COMPONENT_exporters_in_memory_TARGETS
    opentelemetry-cpp::in_memory_span_exporter
    opentelemetry-cpp::in_memory_metric_exporter
)

# COMPONENT exporters_prometheus
set(COMPONENT_exporters_prometheus_TARGETS
    opentelemetry-cpp::prometheus_exporter
)

# COMPONENT exporters_zipkin
set(COMPONENT_exporters_zipkin_TARGETS
    opentelemetry-cpp::zipkin_trace_exporter
)



#-----------------------------------------------------------------------
# COMPONENT to COMPONENT dependencies
#-----------------------------------------------------------------------

# COMPONENT api internal dependencies
set(COMPONENT_api_COMPONENT_DEPENDS
)

# COMPONENT sdk internal dependencies
set(COMPONENT_sdk_COMPONENT_DEPENDS
    api
)

# COMPONENT ext_common internal dependencies
set(COMPONENT_ext_common_COMPONENT_DEPENDS
    api
)

# COMPONENT ext_http_curl internal dependencies
set(COMPONENT_ext_http_curl_COMPONENT_DEPENDS
    api
    sdk
    ext_common
)

# COMPONENT exporters_otlp_common internal dependencies
set(COMPONENT_exporters_otlp_common_COMPONENT_DEPENDS
    api
    sdk
)

# COMPONENT exporters_otlp_grpc internal dependencies
set(COMPONENT_exporters_otlp_grpc_COMPONENT_DEPENDS
    api
    sdk
    ext_common
    exporters_otlp_common
)

# COMPONENT exporters_otlp_http internal dependencies
set(COMPONENT_exporters_otlp_http_COMPONENT_DEPENDS
    api
    sdk
    ext_common
    exporters_otlp_common
    ext_http_curl
)

# COMPONENT exporters_ostream internal dependencies
set(COMPONENT_exporters_ostream_COMPONENT_DEPENDS
    api
    sdk
)

# COMPONENT exporters_in_memory internal dependencies
set(COMPONENT_exporters_in_memory_COMPONENT_DEPENDS
    api
    sdk
)

# COMPONENT exporters_prometheus internal dependencies
set(COMPONENT_exporters_prometheus_COMPONENT_DEPENDS
    api
    sdk
)

# COMPONENT exporters_zipkin internal dependencies
set(COMPONENT_exporters_zipkin_COMPONENT_DEPENDS
    api
    sdk
    ext_common
    ext_http_curl
)




#-----------------------------------------------------------------------
# COMPONENT to THIRDPARTY dependencies
#-----------------------------------------------------------------------

# COMPONENT api thirdparty dependencies
set(COMPONENT_api_THIRDPARTY_DEPENDS
)

# COMPONENT sdk thirdparty dependencies
set(COMPONENT_sdk_THIRDPARTY_DEPENDS
    Threads
)

# COMPONENT ext_common thirdparty dependencies
set(COMPONENT_ext_common_THIRDPARTY_DEPENDS
)

# COMPONENT ext_http_curl thirdparty dependencies
set(COMPONENT_ext_http_curl_THIRDPARTY_DEPENDS
    CURL
)

# COMPONENT exporters_otlp_common thirdparty dependencies
set(COMPONENT_exporters_otlp_common_THIRDPARTY_DEPENDS
    Protobuf
)

# COMPONENT exporters_otlp_grpc thirdparty dependencies
set(COMPONENT_exporters_otlp_grpc_THIRDPARTY_DEPENDS
    gRPC
)

# COMPONENT exporters_otlp_http thirdparty dependencies
set(COMPONENT_exporters_otlp_http_THIRDPARTY_DEPENDS
)

# COMPONENT exporters_ostream thirdparty dependencies
set(COMPONENT_exporters_ostream_THIRDPARTY_DEPENDS
)

# COMPONENT exporters_in_memory thirdparty dependencies
set(COMPONENT_exporters_in_memory_THIRDPARTY_DEPENDS
)

# COMPONENT exporters_prometheus thirdparty dependencies
set(COMPONENT_exporters_prometheus_THIRDPARTY_DEPENDS
    prometheus-cpp
)

# COMPONENT exporters_zipkin thirdparty dependencies
set(COMPONENT_exporters_zipkin_THIRDPARTY_DEPENDS
    nlohmann_json
)


