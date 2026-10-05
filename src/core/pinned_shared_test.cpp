// Linux selftest for the optional MAP_SHARED backing of PinnedArena.
#include "strata/core/pinned.hpp"
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

static bool check_pinned(const strata::core::PinnedArena& arena, uint64_t expected, int devices) {
    if (arena.registered_bytes != expected) {
        std::fprintf(stderr, "expected %llu pinned bytes, got %llu: %s\n",
                     (unsigned long long) expected, (unsigned long long) arena.registered_bytes,
                     arena.note.c_str());
        return false;
    }
    // CPU population after registration must still work. Copy from each registered
    // slice on every visible GPU to exercise PORTABLE and async DMA together.
    for (uint64_t i = 0; i < expected; ++i) arena.data()[i] = (uint8_t) (i * 17 + 91);
    for (int gpu = 0; gpu < devices; ++gpu) {
        if (cudaSetDevice(gpu) != cudaSuccess) return false;
        void* dst = nullptr;
        cudaStream_t stream = nullptr;
        if (cudaMalloc(&dst, 4096) != cudaSuccess) return false;
        if (cudaStreamCreate(&stream) != cudaSuccess) { cudaFree(dst); return false; }
        bool ok = true;
        for (uint64_t off = 0; off < expected && ok; off += 4096) {
            unsigned char got[4096];
            ok = cudaMemcpyAsync(dst, arena.data() + off, sizeof got, cudaMemcpyHostToDevice, stream) == cudaSuccess &&
                 cudaStreamSynchronize(stream) == cudaSuccess &&
                 cudaMemcpy(got, dst, sizeof got, cudaMemcpyDeviceToHost) == cudaSuccess &&
                 std::memcmp(got, arena.data() + off, sizeof got) == 0;
        }
        cudaStreamDestroy(stream);
        cudaFree(dst);
        if (!ok) { std::fprintf(stderr, "GPU %d arena copy failed\n", gpu); return false; }
    }
    return cudaSetDevice(0) == cudaSuccess;
}

int main(int argc, char** argv) {
    const bool require_pinned = argc > 1 && std::string(argv[1]) == "--require-pinned";
    int devices = 0;
    if (require_pinned) {
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
        for (int gpu = 0; gpu < devices; ++gpu) {
            int supported = 0;
            if (cudaDeviceGetAttribute(&supported, cudaDevAttrHostRegisterReadOnlySupported, gpu) != cudaSuccess ||
                !supported) return 77;
        }
#endif
        // Isolate the test from a user's arena pin/lock policy.
        unsetenv("STRATA_ARENA_PIN_GIB");
        setenv("STRATA_ARENA_LOCK", "0", 1);
    }
    std::string path = std::string(argc > 2 ? argv[2] : "/tmp") + "/strata-shared-arena-XXXXXX";
    const int fd = mkstemp(path.data());
    if (fd < 0) {
        std::perror("mkstemp");
        return 2;
    }
    close(fd);

    int fail = 0;
    const uint64_t bytes = 64u << 10;
    const uint64_t pack_hash = 0x0123456789abcdefull;
    const std::vector<uint64_t> no_bounds;
    {
        strata::core::PinnedArena first(bytes, no_bounds, 0, path, pack_hash);
        if (!first.valid()) {
            std::fprintf(stderr, "first shared arena failed: %s\n", first.note.c_str());
            fail = 1;
        } else {
            if (require_pinned && !check_pinned(first, bytes, devices)) fail = 1;
            first.data()[123] = 0x5a;

            strata::core::PinnedArena second(bytes, no_bounds, 0, path, pack_hash);
            if (!second.valid()) {
                std::fprintf(stderr, "second shared arena failed: %s\n", second.note.c_str());
                fail = 1;
            } else if (second.data()[123] != 0x5a) {
                std::fprintf(stderr, "shared mapping did not expose the same bytes\n");
                fail = 1;
            }
            if (require_pinned && second.valid() && !check_pinned(second, bytes, devices)) fail = 1;

            strata::core::PinnedArena wrong_pack(bytes, no_bounds, 0, path, pack_hash + 1);
            if (wrong_pack.valid() || wrong_pack.note.find("pack hash") == std::string::npos) {
                std::fprintf(stderr, "pack mismatch was not refused: %s\n", wrong_pack.note.c_str());
                fail = 1;
            }

            strata::core::PinnedArena wrong_size(bytes * 2, no_bounds, 0, path, pack_hash);
            if (wrong_size.valid() || wrong_size.note.find("expected") == std::string::npos) {
                std::fprintf(stderr, "size mismatch was not refused: %s\n", wrong_size.note.c_str());
                fail = 1;
            }
        }
    }

    if (require_pinned) {
        // Force the per-layer registration path without depending on an OOM.
        const std::vector<uint64_t> bounds{0, bytes / 4, bytes / 2, bytes};
        strata::core::PinnedArena sliced(bytes, bounds, bytes / 2, path, pack_hash);
        if (!sliced.valid() || sliced.registered_slices != 2 || !check_pinned(sliced, bytes / 2, devices)) {
            std::fprintf(stderr, "sliced registration failed: %s\n", sliced.note.c_str());
            fail = 1;
        }
    }

    if (unlink(path.c_str()) != 0) {
        std::perror("unlink");
        fail = 1;
    }
    std::printf("pinned_shared_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
