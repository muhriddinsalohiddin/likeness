// Thin C layer over the ONNX Runtime C API. The runtime library is loaded at
// run time with dlopen/LoadLibrary, so the binary has no link-time dependency
// on it. Every function returns NULL on success or a malloc'ed error message
// that the caller releases with ortx_free.
#ifndef LIKENESS_ORTX_H
#define LIKENESS_ORTX_H

#include <stddef.h>
#include <stdint.h>

#include "onnxruntime_c_api.h"

typedef struct {
  int graph_optimization;  // GraphOptimizationLevel
  int log_severity;        // 0 verbose .. 4 fatal
  int global_pool;         // 1: use the environment's thread pools
  int intra_threads;       // per-session pool size when global_pool is 0
  int spin;                // per-session pool spinning when global_pool is 0
} ortx_session_config;

char* ortx_load(const char* path, uint32_t api_version);
const char* ortx_version(void);

char* ortx_env_create(int log_severity, int global_pool, int intra_threads, int inter_threads, int spin, OrtEnv** env);
void ortx_env_release(OrtEnv* env);

char* ortx_session_create(OrtEnv* env, const void* model, size_t model_len, const ortx_session_config* cfg, OrtSession** out);
void ortx_session_release(OrtSession* s);
char* ortx_session_count(OrtSession* s, size_t* inputs, size_t* outputs);
char* ortx_session_info(OrtSession* s, int output, size_t index, char** name, int* elem_type, int64_t** dims, size_t* ndims);

char* ortx_tensor_create(int elem_type, const int64_t* dims, size_t ndims, size_t nbytes, void** data, OrtValue** out);
char* ortx_value_read(OrtValue* v, int* elem_type, int64_t** dims, size_t* ndims, void** data, size_t* count);
void ortx_value_release(OrtValue* v);

char* ortx_run(OrtSession* s, char** input_names, OrtValue** inputs, size_t ninputs, char** output_names, OrtValue** outputs, size_t noutputs);

void ortx_free(void* p);
void ortx_free_aligned(void* p);

#endif
