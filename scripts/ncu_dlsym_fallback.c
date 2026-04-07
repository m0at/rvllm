#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static void *(*real_dlsym_fn)(void *, const char *) = NULL;
static void *real_libcuda = NULL;

static void init_fallback(void) {
    if (!real_dlsym_fn) {
        real_dlsym_fn = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }
    if (!real_libcuda) {
        real_libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    }
}

void *dlsym(void *handle, const char *symbol) {
    void *ptr;

    init_fallback();
    if (!real_dlsym_fn) {
        return NULL;
    }

    dlerror();
    ptr = real_dlsym_fn(handle, symbol);
    if (ptr || !symbol || symbol[0] != 'c' || symbol[1] != 'u') {
        return ptr;
    }

    if (!real_libcuda) {
        return NULL;
    }

    dlerror();
    ptr = real_dlsym_fn(real_libcuda, symbol);
    return ptr;
}
