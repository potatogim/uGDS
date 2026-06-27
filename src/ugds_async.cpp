#include "ugds_internal.h"
#include <mutex>

/* Backend-neutral async IO logic.
 * The GPU stream callback dispatch (cudaLaunchHostFunc / hipLaunchHostFunc)
 * is in the backend-specific translation units below.
 *
 * This file provides the common validation and request setup,
 * plus the shared callback function. */

static void async_io_callback(void* userData)
{
    AsyncRequest* req = static_cast<AsyncRequest*>(userData);
    size_t size = *req->size_p;
    off_t file_offset = *req->file_offset_p;
    off_t bufPtr_offset = *req->bufPtr_offset_p;

    ssize_t ret = do_io_internal(req->fh, req->bufPtr_base, size,
                                  file_offset, bufPtr_offset, req->opcode);
    *req->bytes_done_p = ret;
    delete req;
}

/* Validate and create async request */
static uGDSError_t async_validate(uGDSHandle_t fh, void* bufPtr_base,
                                   size_t* size_p, off_t* file_offset_p,
                                   off_t* bufPtr_offset_p, ssize_t* bytes_done_p)
{
    if (!g_driver.initialized)
        return make_error(UGDS_DRIVER_NOT_INITIALIZED);
    if (fh == nullptr || bufPtr_base == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    if (size_p == nullptr || file_offset_p == nullptr ||
        bufPtr_offset_p == nullptr || bytes_done_p == nullptr)
        return make_error(UGDS_INVALID_VALUE);

    std::lock_guard<std::mutex> drv_lock(g_driver.lock);
    if (g_driver.buf_registry.find(bufPtr_base) == g_driver.buf_registry.end())
        return make_error(UGDS_INVALID_VALUE);

    return UGDS_OK;
}

static AsyncRequest* make_async_request(uGDSHandle_t fh, void* bufPtr_base,
                                         size_t* size_p, off_t* file_offset_p,
                                         off_t* bufPtr_offset_p, ssize_t* bytes_done_p,
                                         uint8_t opcode)
{
    return new AsyncRequest{
        fh, bufPtr_base, size_p, file_offset_p, bufPtr_offset_p,
        bytes_done_p, opcode
    };
}

/* ── Backend dispatch ── */

/* Forward: launch host function on stream (backend-specific) */
static uGDSError_t async_launch_host_func(cudaStream_t stream,
                                           AsyncRequest* req, uint8_t opcode);

extern "C" uGDSError_t uGDSReadAsync(uGDSHandle_t fh, void* bufPtr_base,
                                       size_t* size_p, off_t* file_offset_p,
                                       off_t* bufPtr_offset_p, ssize_t* bytes_read_p,
                                       cudaStream_t stream)
{
    uGDSError_t st = async_validate(fh, bufPtr_base, size_p, file_offset_p,
                                     bufPtr_offset_p, bytes_read_p);
    if (st.err != UGDS_SUCCESS) return st;

    AsyncRequest* req = make_async_request(fh, bufPtr_base, size_p, file_offset_p,
                                            bufPtr_offset_p, bytes_read_p, NVM_IO_READ);
    return async_launch_host_func(stream, req, NVM_IO_READ);
}

extern "C" uGDSError_t uGDSWriteAsync(uGDSHandle_t fh, void* bufPtr_base,
                                        size_t* size_p, off_t* file_offset_p,
                                        off_t* bufPtr_offset_p, ssize_t* bytes_written_p,
                                        cudaStream_t stream)
{
    uGDSError_t st = async_validate(fh, bufPtr_base, size_p, file_offset_p,
                                     bufPtr_offset_p, bytes_written_p);
    if (st.err != UGDS_SUCCESS) return st;

    AsyncRequest* req = make_async_request(fh, bufPtr_base, size_p, file_offset_p,
                                            bufPtr_offset_p, bytes_written_p, NVM_IO_WRITE);
    return async_launch_host_func(stream, req, NVM_IO_WRITE);
}

extern "C" uGDSError_t uGDSStreamRegister(cudaStream_t stream)
{
    (void)stream;
    return UGDS_OK;
}

extern "C" uGDSError_t uGDSStreamDeregister(cudaStream_t stream)
{
    (void)stream;
    return UGDS_OK;
}

/* ── Backend-specific host function launch ──
 * In dual-backend builds (_CUDA + __HIP_PLATFORM_AMD__), HIP takes
 * priority because the HIP compatibility layer maps cuda* names to
 * hip* equivalents, and the stream passed by HIP callers is a
 * hipStream_t (which cudaLaunchHostFunc would not understand). */

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime_api.h>

static uGDSError_t async_launch_host_func(cudaStream_t stream,
                                           AsyncRequest* req, uint8_t opcode)
{
    (void)opcode;
    hipError_t err = hipLaunchHostFunc(stream, async_io_callback, req);
    if (err != hipSuccess) {
        delete req;
        uGDSError_t e;
        e.err = UGDS_CUDA_DRIVER_ERROR;
        e.cu_err = static_cast<int>(err);
        return e;
    }
    return UGDS_OK;
}

#elif defined(_CUDA) || defined(__CUDACC__)
#include <cuda_runtime.h>

static uGDSError_t async_launch_host_func(cudaStream_t stream,
                                           AsyncRequest* req, uint8_t opcode)
{
    (void)opcode;
    cudaError_t err = cudaLaunchHostFunc(stream, async_io_callback, req);
    if (err != cudaSuccess) {
        delete req;
        uGDSError_t e;
        e.err = UGDS_CUDA_DRIVER_ERROR;
        e.cu_err = static_cast<int>(err);
        return e;
    }
    return UGDS_OK;
}

#elif defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime_api.h>

static uGDSError_t async_launch_host_func(hipStream_t stream,
                                           AsyncRequest* req, uint8_t opcode)
{
    (void)opcode;
    hipError_t err = hipLaunchHostFunc(stream, async_io_callback, req);
    if (err != hipSuccess) {
        delete req;
        uGDSError_t e;
        e.err = UGDS_CUDA_DRIVER_ERROR;
        e.cu_err = static_cast<int>(err);
        return e;
    }
    return UGDS_OK;
}

#else
/* No GPU backend: async IO not available */
static uGDSError_t async_launch_host_func(cudaStream_t stream,
                                           AsyncRequest* req, uint8_t opcode)
{
    (void)stream; (void)opcode;
    delete req;
    return make_error(UGDS_IO_NOT_SUPPORTED);
}
#endif
