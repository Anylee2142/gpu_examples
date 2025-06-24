#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>           // open()
#include <unistd.h>          // close()
#include <sys/mman.h>        // mmap()
#include <string.h>          // memcpy()
#include <time.h>            // clock_gettime()
#include <errno.h>
#include <pthread.h>         // pthreads

#define REGION_SIZE  (128ULL << 30)    // 128 GiB total region
#define CHUNK_SIZE   ( 32ULL << 30)    // 32 GiB per chip slice
#define TEST_SIZE    ( 10ULL << 30)    // 10 GiB write per test

// Measure memcpy from `src` to `dst`; returns seconds elapsed.
static double measure_write(void *dst, const void *src, size_t size) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    memcpy(dst, src, size);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
}

// Thread arg struct
typedef struct {
    void   *dst;
    void   *buf;
    size_t  size;
    double  duration;
} thread_arg_t;

// Thread func: does one measure_write
static void *thread_func(void *varg) {
    thread_arg_t *arg = varg;
    arg->duration = measure_write(arg->dst, arg->buf, arg->size);
    return NULL;
}

int main() {
    const char *dev_path = "/dev/dax0.0";  // adjust to your CXL DAX device
    int fd = open(dev_path, O_RDWR | O_SYNC);
    if (fd < 0) { perror("open"); return 1; }

    // Map the 128 GiB region
    void *base = mmap(NULL, REGION_SIZE, PROT_WRITE|PROT_READ,
                      MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    // Allocate host buffer aligned to 4 KiB
    void *buf;
    posix_memalign(&buf, 4096, TEST_SIZE);
    memset(buf, 0xAB, TEST_SIZE);

    // ─── Scenario 1 (async): two parallel writes into same slice at offsets 0 & 10 GiB ───
    pthread_t s1_th0, s1_th1;
    thread_arg_t s1_arg0 = { base,                     buf, TEST_SIZE, 0.0 };
    thread_arg_t s1_arg1 = { (char*)base + TEST_SIZE,  buf, TEST_SIZE, 0.0 };  // +10 GiB
    struct timespec s1_g0, s1_g1;

    clock_gettime(CLOCK_MONOTONIC_RAW, &s1_g0);
    pthread_create(&s1_th0, NULL, thread_func, &s1_arg0);
    pthread_create(&s1_th1, NULL, thread_func, &s1_arg1);
    pthread_join(s1_th0, NULL);
    pthread_join(s1_th1, NULL);
    clock_gettime(CLOCK_MONOTONIC_RAW, &s1_g1);

    double s1_wall = (s1_g1.tv_sec - s1_g0.tv_sec)
                   + (s1_g1.tv_nsec - s1_g0.tv_nsec)/1e9;
    printf("Scenario 1 (async): writes at offsets 0 & 10 GiB:\n"
           "  thread0 = %.3f s, thread1 = %.3f s, wall-clock = %.3f s\n\n",
           s1_arg0.duration, s1_arg1.duration, s1_wall);

    // ─── Scenario 2 (async): two parallel writes to slice 0 & slice 1 ───
    pthread_t s2_th0, s2_th1;
    thread_arg_t s2_arg0 = { base,                    buf, TEST_SIZE, 0.0 };
    thread_arg_t s2_arg1 = { (char*)base + CHUNK_SIZE, buf, TEST_SIZE, 0.0 };
    struct timespec s2_g0, s2_g1;

    clock_gettime(CLOCK_MONOTONIC_RAW, &s2_g0);
    pthread_create(&s2_th0, NULL, thread_func, &s2_arg0);
    pthread_create(&s2_th1, NULL, thread_func, &s2_arg1);
    pthread_join(s2_th0, NULL);
    pthread_join(s2_th1, NULL);
    clock_gettime(CLOCK_MONOTONIC_RAW, &s2_g1);

    double s2_wall = (s2_g1.tv_sec - s2_g0.tv_sec)
                   + (s2_g1.tv_nsec - s2_g0.tv_nsec)/1e9;
    printf("Scenario 2 (async): writes to slice 0 (0–32 GiB) & slice 1 (32–64 GiB):\n"
           "  thread0 = %.3f s, thread1 = %.3f s, wall-clock = %.3f s\n",
           s2_arg0.duration, s2_arg1.duration, s2_wall);

    // Cleanup
    free(buf);
    munmap(base, REGION_SIZE);
    close(fd);
    return 0;
}
# gcc -O3 -march=native -pthread -o cxl_both_async cxl_both_async.c -lrt
# ./cxl_both_async
