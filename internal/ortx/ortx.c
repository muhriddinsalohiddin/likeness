#include "ortx.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static const OrtApiBase* base;
static const OrtApi* api;

static char* errorf(const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  return strdup(buf);
}

// status_error converts an OrtStatus into an owned message, or NULL.
static char* status_error(OrtStatus* st) {
  if (st == NULL) {
    return NULL;
  }
  const char* msg = api->GetErrorMessage(st);
  char* out = strdup(msg != NULL ? msg : "onnxruntime error");
  api->ReleaseStatus(st);
  return out;
}

char* ortx_load(const char* path, uint32_t api_version) {
  if (api != NULL) {
    return NULL;
  }
  const OrtApiBase* (*get_base)(void) = NULL;
#ifdef _WIN32
  HMODULE h = LoadLibraryA(path);
  if (h == NULL) {
    return errorf("LoadLibrary(%s) failed with error %lu", path, GetLastError());
  }
  get_base = (const OrtApiBase * (*)(void)) GetProcAddress(h, "OrtGetApiBase");
#else
  void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (h == NULL) {
    const char* why = dlerror();
    return errorf("%s", why != NULL ? why : "dlopen failed");
  }
  *(void**)(&get_base) = dlsym(h, "OrtGetApiBase");
#endif
  if (get_base == NULL) {
    return errorf("%s does not export OrtGetApiBase: not an onnxruntime library", path);
  }
  base = get_base();
  const OrtApi* a = base->GetApi(api_version);
  if (a == NULL) {
    return errorf("onnxruntime %s is too old: it does not provide C API version %u",
                  base->GetVersionString(), api_version);
  }
  api = a;
  return NULL;
}

const char* ortx_version(void) {
  return base != NULL ? base->GetVersionString() : "";
}

char* ortx_env_create(int log_severity, int global_pool, int intra_threads, int inter_threads, int spin, OrtEnv** env) {
  OrtStatus* st;
  if (!global_pool) {
    st = api->CreateEnv((OrtLoggingLevel)log_severity, "likeness", env);
    return status_error(st);
  }
  OrtThreadingOptions* tp = NULL;
  if ((st = api->CreateThreadingOptions(&tp)) != NULL) {
    return status_error(st);
  }
  if ((st = api->SetGlobalIntraOpNumThreads(tp, intra_threads)) == NULL &&
      (st = api->SetGlobalInterOpNumThreads(tp, inter_threads)) == NULL &&
      (st = api->SetGlobalSpinControl(tp, spin)) == NULL) {
    st = api->CreateEnvWithGlobalThreadPools((OrtLoggingLevel)log_severity, "likeness", tp, env);
  }
  api->ReleaseThreadingOptions(tp);
  return status_error(st);
}

void ortx_env_release(OrtEnv* env) {
  if (env != NULL) {
    api->ReleaseEnv(env);
  }
}

char* ortx_session_create(OrtEnv* env, const void* model, size_t model_len, const ortx_session_config* cfg, OrtSession** out) {
  OrtSessionOptions* so = NULL;
  OrtStatus* st = api->CreateSessionOptions(&so);
  if (st != NULL) {
    return status_error(st);
  }
  if ((st = api->SetSessionGraphOptimizationLevel(so, (GraphOptimizationLevel)cfg->graph_optimization)) == NULL &&
      (st = api->SetSessionExecutionMode(so, ORT_SEQUENTIAL)) == NULL &&
      (st = api->SetSessionLogSeverityLevel(so, cfg->log_severity)) == NULL) {
    if (cfg->global_pool) {
      st = api->DisablePerSessionThreads(so);
    } else if ((st = api->SetIntraOpNumThreads(so, cfg->intra_threads)) == NULL &&
               (st = api->SetInterOpNumThreads(so, 1)) == NULL) {
      st = api->AddSessionConfigEntry(so, "session.intra_op.allow_spinning", cfg->spin ? "1" : "0");
    }
  }
  if (st == NULL) {
    st = api->CreateSessionFromArray(env, model, model_len, so, out);
  }
  api->ReleaseSessionOptions(so);
  return status_error(st);
}

void ortx_session_release(OrtSession* s) {
  if (s != NULL) {
    api->ReleaseSession(s);
  }
}

char* ortx_session_count(OrtSession* s, size_t* inputs, size_t* outputs) {
  OrtStatus* st = api->SessionGetInputCount(s, inputs);
  if (st == NULL) {
    st = api->SessionGetOutputCount(s, outputs);
  }
  return status_error(st);
}

char* ortx_session_info(OrtSession* s, int output, size_t index, char** name, int* elem_type, int64_t** dims, size_t* ndims) {
  OrtAllocator* alloc = NULL;
  OrtStatus* st = api->GetAllocatorWithDefaultOptions(&alloc);
  if (st != NULL) {
    return status_error(st);
  }
  char* n = NULL;
  st = output ? api->SessionGetOutputName(s, index, alloc, &n) : api->SessionGetInputName(s, index, alloc, &n);
  if (st != NULL) {
    return status_error(st);
  }
  *name = strdup(n);
  OrtStatus* freed = api->AllocatorFree(alloc, n);
  if (freed != NULL) {
    api->ReleaseStatus(freed);
  }

  OrtTypeInfo* ti = NULL;
  st = output ? api->SessionGetOutputTypeInfo(s, index, &ti) : api->SessionGetInputTypeInfo(s, index, &ti);
  if (st != NULL) {
    return status_error(st);
  }
  const OrtTensorTypeAndShapeInfo* tsi = NULL;
  *elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  *dims = NULL;
  *ndims = 0;
  if ((st = api->CastTypeInfoToTensorInfo(ti, &tsi)) == NULL && tsi != NULL) {
    ONNXTensorElementDataType t;
    if ((st = api->GetTensorElementType(tsi, &t)) == NULL && (st = api->GetDimensionsCount(tsi, ndims)) == NULL) {
      *elem_type = (int)t;
      if (*ndims > 0) {
        *dims = (int64_t*)malloc(*ndims * sizeof(int64_t));
        st = api->GetDimensions(tsi, *dims, *ndims);
      }
    }
  }
  api->ReleaseTypeInfo(ti);
  return status_error(st);
}

static void* alloc_aligned(size_t nbytes) {
  // Tensors like 64-byte alignment; zeroed so padding never carries garbage.
  size_t n = (nbytes + 63) & ~(size_t)63;
  if (n == 0) {
    n = 64;
  }
#ifdef _WIN32
  void* p = _aligned_malloc(n, 64);
#else
  void* p = NULL;
  if (posix_memalign(&p, 64, n) != 0) {
    p = NULL;
  }
#endif
  if (p != NULL) {
    memset(p, 0, n);
  }
  return p;
}

void ortx_free_aligned(void* p) {
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}

char* ortx_tensor_create(int elem_type, const int64_t* dims, size_t ndims, size_t nbytes, void** data, OrtValue** out) {
  void* p = alloc_aligned(nbytes);
  if (p == NULL) {
    return errorf("out of memory allocating a %zu byte tensor", nbytes);
  }
  OrtMemoryInfo* mi = NULL;
  OrtStatus* st = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi);
  if (st == NULL) {
    st = api->CreateTensorWithDataAsOrtValue(mi, p, nbytes, dims, ndims, (ONNXTensorElementDataType)elem_type, out);
    api->ReleaseMemoryInfo(mi);
  }
  if (st != NULL) {
    ortx_free_aligned(p);
    return status_error(st);
  }
  *data = p;
  return NULL;
}

char* ortx_value_read(OrtValue* v, int* elem_type, int64_t** dims, size_t* ndims, void** data, size_t* count) {
  OrtTensorTypeAndShapeInfo* info = NULL;
  OrtStatus* st = api->GetTensorTypeAndShape(v, &info);
  if (st != NULL) {
    return status_error(st);
  }
  ONNXTensorElementDataType t;
  *dims = NULL;
  if ((st = api->GetTensorElementType(info, &t)) == NULL &&
      (st = api->GetDimensionsCount(info, ndims)) == NULL &&
      (st = api->GetTensorShapeElementCount(info, count)) == NULL) {
    *elem_type = (int)t;
    if (*ndims > 0) {
      *dims = (int64_t*)malloc(*ndims * sizeof(int64_t));
      st = api->GetDimensions(info, *dims, *ndims);
    }
  }
  api->ReleaseTensorTypeAndShapeInfo(info);
  if (st == NULL) {
    st = api->GetTensorMutableData(v, data);
  }
  return status_error(st);
}

void ortx_value_release(OrtValue* v) {
  if (v != NULL) {
    api->ReleaseValue(v);
  }
}

char* ortx_run(OrtSession* s, char** input_names, OrtValue** inputs, size_t ninputs, char** output_names, OrtValue** outputs, size_t noutputs) {
  return status_error(api->Run(s, NULL, (const char* const*)input_names, (const OrtValue* const*)inputs, ninputs,
                               (const char* const*)output_names, noutputs, outputs));
}

void ortx_free(void* p) {
  free(p);
}
