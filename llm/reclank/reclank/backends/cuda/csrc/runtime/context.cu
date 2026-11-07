/* The context released (docs/GPU_SWAP_PLAN.md 3.6). rc_gpu_shutdown: with no program loaded and no
 * block of rc_gpu_malloc left (the host freed them: RESIDENCY's release), the copy workers and
 * the helper idle, it waits for the GPU, resets the device (cudaDeviceReset: the context, its
 * memory, streams, events, graphs and registrations go) and adds 1 to the epoch. The next CUDA call
 * makes a new context; what the library keeps of a context is made again where its epoch is old
 * (gg_stream_now, the workers, the fences, the kernels' attributes), and gg_context_up registers
 * the host memory again and sets the host's symbols. The expert device (runtime/expert_dev.cu, a
 * second GPU) and cudaMallocHost's memory (a device without host pointers: rc_gpu_host_alloc)
 * keep the context: -1 then, with the reason in gg_error. */
extern "C" int rc_gpu_shutdown(void)
{
    if (gg_nprog > 0) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_shutdown: %d programs still loaded", gg_nprog);
        return -1;
    }
    int64_t nm = __atomic_load_n(&gg_nmalloc, __ATOMIC_RELAXED);
    if (nm > 0) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_shutdown: %lld device blocks still held", (long long)nm);
        return -1;
    }
    if (__atomic_load_n(&gg_nmallochost, __ATOMIC_RELAXED) > 0) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_shutdown: pinned memory of cudaMallocHost (no host pointers)");
        return -1;
    }
    if (ex_own != NULL) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_shutdown: the expert device is open");
        return -1;
    }
    /* the workers idle: their queues empty, the helper's job done */
    pthread_mutex_lock(&gg_mu);
    while (gg_job_head != gg_job_tail) {
        pthread_cond_wait(&gg_cv, &gg_mu);
    }
    pthread_mutex_unlock(&gg_mu);
    pthread_mutex_lock(&gg_cmu);
    while (gg_cjob_head != gg_cjob_tail) {
        pthread_cond_wait(&gg_ccv, &gg_cmu);
    }
    pthread_mutex_unlock(&gg_cmu);
    pthread_mutex_lock(&gg_hmu);
    while (gg_hjob != NULL || gg_hbusy) {
        pthread_cond_wait(&gg_hcv, &gg_hmu);
    }
    pthread_mutex_unlock(&gg_hmu);
    CK(cudaDeviceSynchronize());
    CK(cudaDeviceReset());
    ++gg_epoch;
    gg_main_stream = NULL;
    gg_lane_stream = NULL;
    gg_tstream = NULL;              /* (this thread's; another lane's sees the epoch) */
    gg_pool_tab = NULL;             /* (the device's: gone with the context) */
    gg_pool_cap = 0;
    e3_lib_ws = NULL;
    e3_lib_ws_n = 0;
    return 0;
}

/* The epoch (1 until the first rc_gpu_shutdown), and the seconds the last registrations again took. */
extern "C" int rc_gpu_epoch(double *reg_secs)
{
    if (reg_secs != NULL) {
        *reg_secs = gg_reg_secs;
    }
    return gg_epoch;
}
