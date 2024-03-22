void gemm_atomic_f64f64f64f64f64_An_Bt_Md_Nd_Kd_Astride1_d_Bstride1_d_Cstride1_d_alphad_beta3ff0000000000000(
    uint M, uint N, uint K, double alpha, global double *A, uint A_stride,
    uint A_stride1, global double *B, uint B_stride, uint B_stride1,
    double beta, global double *C, uint C_stride, uint C_stride1) {
  uint m = get_sub_group_local_id();
  double c[16];
  uint sg_n = get_sub_group_id();
  uint blocks = 1 + (N - 1) / 16u;
  blocks = 1 + (blocks - 1);
  uint bs = N / blocks;
  uint bs_1 = bs + 1;
  uint rem = N % blocks;
  uint blck;
  __attribute__((opencl_unroll_hint(1))) for (blck = bs_1 * sg_n;
                                              blck < bs_1 * rem; blck += bs_1) {
    global double *Bb = B + blck;
    global double *Cb = C + C_stride1 * blck;
    uint sg_m = 0;
    uint blocks1 = M / 16u;
    uint rem1 = M % 16u;
    uint blck1;
    __attribute__((opencl_unroll_hint(1))) for (blck1 = 16u * sg_m;
                                                blck1 < 16u * blocks1;
                                                blck1 += 16u) {
      global double *Ab = A + blck1;
      global double *Cb1 = Cb + blck1;
      global double *Ab1 = Ab;
      global double *Bb1 = Bb;
      __attribute__((opencl_unroll_hint(16))) for (short n = 0; n < 16; ++n) {
        c[n] = 0x0p+0;
      }
      uint KmultipleKb = K / 8 * 8;
      __attribute__((opencl_unroll_hint(1))) for (short kb = 0;
                                                  kb < KmultipleKb; kb += 8) {
        double a[8];
        double b[8];
        a[0u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[1u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[2u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[3u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[4u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[5u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[6u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[7u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        if (m < bs_1) {
          b[0] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs_1) {
          b[1] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs_1) {
          b[2] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs_1) {
          b[3] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs_1) {
          b[4] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs_1) {
          b[5] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs_1) {
          b[6] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs_1) {
          b[7] = Bb1[m];
        }
        Bb1 += B_stride1;
        c[0] = fma(a[0u], sub_group_broadcast(b[0], 0), c[0]);
        c[1] = fma(a[0u], sub_group_broadcast(b[0], 1), c[1]);
        c[2] = fma(a[0u], sub_group_broadcast(b[0], 2), c[2]);
        c[3] = fma(a[0u], sub_group_broadcast(b[0], 3), c[3]);
        c[0] = fma(a[1u], sub_group_broadcast(b[1], 0), c[0]);
        c[1] = fma(a[1u], sub_group_broadcast(b[1], 1), c[1]);
        c[2] = fma(a[1u], sub_group_broadcast(b[1], 2), c[2]);
        c[3] = fma(a[1u], sub_group_broadcast(b[1], 3), c[3]);
        c[0] = fma(a[2u], sub_group_broadcast(b[2], 0), c[0]);
        c[1] = fma(a[2u], sub_group_broadcast(b[2], 1), c[1]);
        c[2] = fma(a[2u], sub_group_broadcast(b[2], 2), c[2]);
        c[3] = fma(a[2u], sub_group_broadcast(b[2], 3), c[3]);
        c[0] = fma(a[3u], sub_group_broadcast(b[3], 0), c[0]);
        c[1] = fma(a[3u], sub_group_broadcast(b[3], 1), c[1]);
        c[2] = fma(a[3u], sub_group_broadcast(b[3], 2), c[2]);
        c[3] = fma(a[3u], sub_group_broadcast(b[3], 3), c[3]);
        c[0] = fma(a[4u], sub_group_broadcast(b[4], 0), c[0]);
        c[1] = fma(a[4u], sub_group_broadcast(b[4], 1), c[1]);
        c[2] = fma(a[4u], sub_group_broadcast(b[4], 2), c[2]);
        c[3] = fma(a[4u], sub_group_broadcast(b[4], 3), c[3]);
        c[0] = fma(a[5u], sub_group_broadcast(b[5], 0), c[0]);
        c[1] = fma(a[5u], sub_group_broadcast(b[5], 1), c[1]);
        c[2] = fma(a[5u], sub_group_broadcast(b[5], 2), c[2]);
        c[3] = fma(a[5u], sub_group_broadcast(b[5], 3), c[3]);
        c[0] = fma(a[6u], sub_group_broadcast(b[6], 0), c[0]);
        c[1] = fma(a[6u], sub_group_broadcast(b[6], 1), c[1]);
        c[2] = fma(a[6u], sub_group_broadcast(b[6], 2), c[2]);
        c[3] = fma(a[6u], sub_group_broadcast(b[6], 3), c[3]);
        c[0] = fma(a[7u], sub_group_broadcast(b[7], 0), c[0]);
        c[1] = fma(a[7u], sub_group_broadcast(b[7], 1), c[1]);
        c[2] = fma(a[7u], sub_group_broadcast(b[7], 2), c[2]);
        c[3] = fma(a[7u], sub_group_broadcast(b[7], 3), c[3]);
        c[4] = fma(a[0u], sub_group_broadcast(b[0], 4), c[4]);
        c[5] = fma(a[0u], sub_group_broadcast(b[0], 5), c[5]);
        c[6] = fma(a[0u], sub_group_broadcast(b[0], 6), c[6]);
        c[7] = fma(a[0u], sub_group_broadcast(b[0], 7), c[7]);
        c[4] = fma(a[1u], sub_group_broadcast(b[1], 4), c[4]);
        c[5] = fma(a[1u], sub_group_broadcast(b[1], 5), c[5]);
        c[6] = fma(a[1u], sub_group_broadcast(b[1], 6), c[6]);
        c[7] = fma(a[1u], sub_group_broadcast(b[1], 7), c[7]);
        c[4] = fma(a[2u], sub_group_broadcast(b[2], 4), c[4]);
        c[5] = fma(a[2u], sub_group_broadcast(b[2], 5), c[5]);
        c[6] = fma(a[2u], sub_group_broadcast(b[2], 6), c[6]);
        c[7] = fma(a[2u], sub_group_broadcast(b[2], 7), c[7]);
        c[4] = fma(a[3u], sub_group_broadcast(b[3], 4), c[4]);
        c[5] = fma(a[3u], sub_group_broadcast(b[3], 5), c[5]);
        c[6] = fma(a[3u], sub_group_broadcast(b[3], 6), c[6]);
        c[7] = fma(a[3u], sub_group_broadcast(b[3], 7), c[7]);
        c[4] = fma(a[4u], sub_group_broadcast(b[4], 4), c[4]);
        c[5] = fma(a[4u], sub_group_broadcast(b[4], 5), c[5]);
        c[6] = fma(a[4u], sub_group_broadcast(b[4], 6), c[6]);
        c[7] = fma(a[4u], sub_group_broadcast(b[4], 7), c[7]);
        c[4] = fma(a[5u], sub_group_broadcast(b[5], 4), c[4]);
        c[5] = fma(a[5u], sub_group_broadcast(b[5], 5), c[5]);
        c[6] = fma(a[5u], sub_group_broadcast(b[5], 6), c[6]);
        c[7] = fma(a[5u], sub_group_broadcast(b[5], 7), c[7]);
        c[4] = fma(a[6u], sub_group_broadcast(b[6], 4), c[4]);
        c[5] = fma(a[6u], sub_group_broadcast(b[6], 5), c[5]);
        c[6] = fma(a[6u], sub_group_broadcast(b[6], 6), c[6]);
        c[7] = fma(a[6u], sub_group_broadcast(b[6], 7), c[7]);
        c[4] = fma(a[7u], sub_group_broadcast(b[7], 4), c[4]);
        c[5] = fma(a[7u], sub_group_broadcast(b[7], 5), c[5]);
        c[6] = fma(a[7u], sub_group_broadcast(b[7], 6), c[6]);
        c[7] = fma(a[7u], sub_group_broadcast(b[7], 7), c[7]);
        c[8] = fma(a[0u], sub_group_broadcast(b[0], 8), c[8]);
        c[9] = fma(a[0u], sub_group_broadcast(b[0], 9), c[9]);
        c[10] = fma(a[0u], sub_group_broadcast(b[0], 10), c[10]);
        c[11] = fma(a[0u], sub_group_broadcast(b[0], 11), c[11]);
        c[8] = fma(a[1u], sub_group_broadcast(b[1], 8), c[8]);
        c[9] = fma(a[1u], sub_group_broadcast(b[1], 9), c[9]);
        c[10] = fma(a[1u], sub_group_broadcast(b[1], 10), c[10]);
        c[11] = fma(a[1u], sub_group_broadcast(b[1], 11), c[11]);
        c[8] = fma(a[2u], sub_group_broadcast(b[2], 8), c[8]);
        c[9] = fma(a[2u], sub_group_broadcast(b[2], 9), c[9]);
        c[10] = fma(a[2u], sub_group_broadcast(b[2], 10), c[10]);
        c[11] = fma(a[2u], sub_group_broadcast(b[2], 11), c[11]);
        c[8] = fma(a[3u], sub_group_broadcast(b[3], 8), c[8]);
        c[9] = fma(a[3u], sub_group_broadcast(b[3], 9), c[9]);
        c[10] = fma(a[3u], sub_group_broadcast(b[3], 10), c[10]);
        c[11] = fma(a[3u], sub_group_broadcast(b[3], 11), c[11]);
        c[8] = fma(a[4u], sub_group_broadcast(b[4], 8), c[8]);
        c[9] = fma(a[4u], sub_group_broadcast(b[4], 9), c[9]);
        c[10] = fma(a[4u], sub_group_broadcast(b[4], 10), c[10]);
        c[11] = fma(a[4u], sub_group_broadcast(b[4], 11), c[11]);
        c[8] = fma(a[5u], sub_group_broadcast(b[5], 8), c[8]);
        c[9] = fma(a[5u], sub_group_broadcast(b[5], 9), c[9]);
        c[10] = fma(a[5u], sub_group_broadcast(b[5], 10), c[10]);
        c[11] = fma(a[5u], sub_group_broadcast(b[5], 11), c[11]);
        c[8] = fma(a[6u], sub_group_broadcast(b[6], 8), c[8]);
        c[9] = fma(a[6u], sub_group_broadcast(b[6], 9), c[9]);
        c[10] = fma(a[6u], sub_group_broadcast(b[6], 10), c[10]);
        c[11] = fma(a[6u], sub_group_broadcast(b[6], 11), c[11]);
        c[8] = fma(a[7u], sub_group_broadcast(b[7], 8), c[8]);
        c[9] = fma(a[7u], sub_group_broadcast(b[7], 9), c[9]);
        c[10] = fma(a[7u], sub_group_broadcast(b[7], 10), c[10]);
        c[11] = fma(a[7u], sub_group_broadcast(b[7], 11), c[11]);
        c[12] = fma(a[0u], sub_group_broadcast(b[0], 12), c[12]);
        c[13] = fma(a[0u], sub_group_broadcast(b[0], 13), c[13]);
        c[14] = fma(a[0u], sub_group_broadcast(b[0], 14), c[14]);
        c[15] = fma(a[0u], sub_group_broadcast(b[0], 15), c[15]);
        c[12] = fma(a[1u], sub_group_broadcast(b[1], 12), c[12]);
        c[13] = fma(a[1u], sub_group_broadcast(b[1], 13), c[13]);
        c[14] = fma(a[1u], sub_group_broadcast(b[1], 14), c[14]);
        c[15] = fma(a[1u], sub_group_broadcast(b[1], 15), c[15]);
        c[12] = fma(a[2u], sub_group_broadcast(b[2], 12), c[12]);
        c[13] = fma(a[2u], sub_group_broadcast(b[2], 13), c[13]);
        c[14] = fma(a[2u], sub_group_broadcast(b[2], 14), c[14]);
        c[15] = fma(a[2u], sub_group_broadcast(b[2], 15), c[15]);
        c[12] = fma(a[3u], sub_group_broadcast(b[3], 12), c[12]);
        c[13] = fma(a[3u], sub_group_broadcast(b[3], 13), c[13]);
        c[14] = fma(a[3u], sub_group_broadcast(b[3], 14), c[14]);
        c[15] = fma(a[3u], sub_group_broadcast(b[3], 15), c[15]);
        c[12] = fma(a[4u], sub_group_broadcast(b[4], 12), c[12]);
        c[13] = fma(a[4u], sub_group_broadcast(b[4], 13), c[13]);
        c[14] = fma(a[4u], sub_group_broadcast(b[4], 14), c[14]);
        c[15] = fma(a[4u], sub_group_broadcast(b[4], 15), c[15]);
        c[12] = fma(a[5u], sub_group_broadcast(b[5], 12), c[12]);
        c[13] = fma(a[5u], sub_group_broadcast(b[5], 13), c[13]);
        c[14] = fma(a[5u], sub_group_broadcast(b[5], 14), c[14]);
        c[15] = fma(a[5u], sub_group_broadcast(b[5], 15), c[15]);
        c[12] = fma(a[6u], sub_group_broadcast(b[6], 12), c[12]);
        c[13] = fma(a[6u], sub_group_broadcast(b[6], 13), c[13]);
        c[14] = fma(a[6u], sub_group_broadcast(b[6], 14), c[14]);
        c[15] = fma(a[6u], sub_group_broadcast(b[6], 15), c[15]);
        c[12] = fma(a[7u], sub_group_broadcast(b[7], 12), c[12]);
        c[13] = fma(a[7u], sub_group_broadcast(b[7], 13), c[13]);
        c[14] = fma(a[7u], sub_group_broadcast(b[7], 14), c[14]);
        c[15] = fma(a[7u], sub_group_broadcast(b[7], 15), c[15]);
      }
      if (K - KmultipleKb > 0) {
        __attribute__((opencl_unroll_hint(1))) for (short kb = KmultipleKb;
                                                    kb < K; kb += 1) {
          double a[1];
          double b[1];
          a[0u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
          Ab1 += A_stride1;
          if (m < bs_1) {
            b[0] = Bb1[m];
          }
          Bb1 += B_stride1;
          c[0] = fma(a[0u], sub_group_broadcast(b[0], 0), c[0]);
          c[1] = fma(a[0u], sub_group_broadcast(b[0], 1), c[1]);
          c[2] = fma(a[0u], sub_group_broadcast(b[0], 2), c[2]);
          c[3] = fma(a[0u], sub_group_broadcast(b[0], 3), c[3]);
          c[4] = fma(a[0u], sub_group_broadcast(b[0], 4), c[4]);
          c[5] = fma(a[0u], sub_group_broadcast(b[0], 5), c[5]);
          c[6] = fma(a[0u], sub_group_broadcast(b[0], 6), c[6]);
          c[7] = fma(a[0u], sub_group_broadcast(b[0], 7), c[7]);
          c[8] = fma(a[0u], sub_group_broadcast(b[0], 8), c[8]);
          c[9] = fma(a[0u], sub_group_broadcast(b[0], 9), c[9]);
          c[10] = fma(a[0u], sub_group_broadcast(b[0], 10), c[10]);
          c[11] = fma(a[0u], sub_group_broadcast(b[0], 11), c[11]);
          c[12] = fma(a[0u], sub_group_broadcast(b[0], 12), c[12]);
          c[13] = fma(a[0u], sub_group_broadcast(b[0], 13), c[13]);
          c[14] = fma(a[0u], sub_group_broadcast(b[0], 14), c[14]);
          c[15] = fma(a[0u], sub_group_broadcast(b[0], 15), c[15]);
        }
      }
      __attribute__((opencl_unroll_hint(1))) for (short n = 0;
                                                  n < min(bs_1, (uint)16);
                                                  ++n) {
        global double *Cdst = Cb1 + m;
        atomic_fetch_add((global volatile atomic_double *)Cdst, alpha * c[n]);
        Cb1 += (uint)C_stride1;
      }
    }
    if (rem1 > 0) {
      blck1 = blocks1 * 16u;
      if (sg_m == 0u) {
        global double *Ab = A + blck1;
        global double *Cb2 = Cb + blck1;
        global double *Ab1 = Ab;
        global double *Bb2 = Bb;
        __attribute__((opencl_unroll_hint(16))) for (short n = 0; n < 16; ++n) {
          c[n] = 0x0p+0;
        }
        uint KmultipleKb = K / 8 * 8;
        __attribute__((opencl_unroll_hint(1))) for (short kb = 0;
                                                    kb < KmultipleKb; kb += 8) {
          double a[8];
          double b[8];
          if (m < rem1) {
            a[0u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem1) {
            a[1u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem1) {
            a[2u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem1) {
            a[3u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem1) {
            a[4u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem1) {
            a[5u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem1) {
            a[6u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem1) {
            a[7u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < bs_1) {
            b[0] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs_1) {
            b[1] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs_1) {
            b[2] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs_1) {
            b[3] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs_1) {
            b[4] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs_1) {
            b[5] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs_1) {
            b[6] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs_1) {
            b[7] = Bb2[m];
          }
          Bb2 += B_stride1;
          c[0] = fma(a[0u], sub_group_broadcast(b[0], 0), c[0]);
          c[1] = fma(a[0u], sub_group_broadcast(b[0], 1), c[1]);
          c[2] = fma(a[0u], sub_group_broadcast(b[0], 2), c[2]);
          c[3] = fma(a[0u], sub_group_broadcast(b[0], 3), c[3]);
          c[0] = fma(a[1u], sub_group_broadcast(b[1], 0), c[0]);
          c[1] = fma(a[1u], sub_group_broadcast(b[1], 1), c[1]);
          c[2] = fma(a[1u], sub_group_broadcast(b[1], 2), c[2]);
          c[3] = fma(a[1u], sub_group_broadcast(b[1], 3), c[3]);
          c[0] = fma(a[2u], sub_group_broadcast(b[2], 0), c[0]);
          c[1] = fma(a[2u], sub_group_broadcast(b[2], 1), c[1]);
          c[2] = fma(a[2u], sub_group_broadcast(b[2], 2), c[2]);
          c[3] = fma(a[2u], sub_group_broadcast(b[2], 3), c[3]);
          c[0] = fma(a[3u], sub_group_broadcast(b[3], 0), c[0]);
          c[1] = fma(a[3u], sub_group_broadcast(b[3], 1), c[1]);
          c[2] = fma(a[3u], sub_group_broadcast(b[3], 2), c[2]);
          c[3] = fma(a[3u], sub_group_broadcast(b[3], 3), c[3]);
          c[0] = fma(a[4u], sub_group_broadcast(b[4], 0), c[0]);
          c[1] = fma(a[4u], sub_group_broadcast(b[4], 1), c[1]);
          c[2] = fma(a[4u], sub_group_broadcast(b[4], 2), c[2]);
          c[3] = fma(a[4u], sub_group_broadcast(b[4], 3), c[3]);
          c[0] = fma(a[5u], sub_group_broadcast(b[5], 0), c[0]);
          c[1] = fma(a[5u], sub_group_broadcast(b[5], 1), c[1]);
          c[2] = fma(a[5u], sub_group_broadcast(b[5], 2), c[2]);
          c[3] = fma(a[5u], sub_group_broadcast(b[5], 3), c[3]);
          c[0] = fma(a[6u], sub_group_broadcast(b[6], 0), c[0]);
          c[1] = fma(a[6u], sub_group_broadcast(b[6], 1), c[1]);
          c[2] = fma(a[6u], sub_group_broadcast(b[6], 2), c[2]);
          c[3] = fma(a[6u], sub_group_broadcast(b[6], 3), c[3]);
          c[0] = fma(a[7u], sub_group_broadcast(b[7], 0), c[0]);
          c[1] = fma(a[7u], sub_group_broadcast(b[7], 1), c[1]);
          c[2] = fma(a[7u], sub_group_broadcast(b[7], 2), c[2]);
          c[3] = fma(a[7u], sub_group_broadcast(b[7], 3), c[3]);
          c[4] = fma(a[0u], sub_group_broadcast(b[0], 4), c[4]);
          c[5] = fma(a[0u], sub_group_broadcast(b[0], 5), c[5]);
          c[6] = fma(a[0u], sub_group_broadcast(b[0], 6), c[6]);
          c[7] = fma(a[0u], sub_group_broadcast(b[0], 7), c[7]);
          c[4] = fma(a[1u], sub_group_broadcast(b[1], 4), c[4]);
          c[5] = fma(a[1u], sub_group_broadcast(b[1], 5), c[5]);
          c[6] = fma(a[1u], sub_group_broadcast(b[1], 6), c[6]);
          c[7] = fma(a[1u], sub_group_broadcast(b[1], 7), c[7]);
          c[4] = fma(a[2u], sub_group_broadcast(b[2], 4), c[4]);
          c[5] = fma(a[2u], sub_group_broadcast(b[2], 5), c[5]);
          c[6] = fma(a[2u], sub_group_broadcast(b[2], 6), c[6]);
          c[7] = fma(a[2u], sub_group_broadcast(b[2], 7), c[7]);
          c[4] = fma(a[3u], sub_group_broadcast(b[3], 4), c[4]);
          c[5] = fma(a[3u], sub_group_broadcast(b[3], 5), c[5]);
          c[6] = fma(a[3u], sub_group_broadcast(b[3], 6), c[6]);
          c[7] = fma(a[3u], sub_group_broadcast(b[3], 7), c[7]);
          c[4] = fma(a[4u], sub_group_broadcast(b[4], 4), c[4]);
          c[5] = fma(a[4u], sub_group_broadcast(b[4], 5), c[5]);
          c[6] = fma(a[4u], sub_group_broadcast(b[4], 6), c[6]);
          c[7] = fma(a[4u], sub_group_broadcast(b[4], 7), c[7]);
          c[4] = fma(a[5u], sub_group_broadcast(b[5], 4), c[4]);
          c[5] = fma(a[5u], sub_group_broadcast(b[5], 5), c[5]);
          c[6] = fma(a[5u], sub_group_broadcast(b[5], 6), c[6]);
          c[7] = fma(a[5u], sub_group_broadcast(b[5], 7), c[7]);
          c[4] = fma(a[6u], sub_group_broadcast(b[6], 4), c[4]);
          c[5] = fma(a[6u], sub_group_broadcast(b[6], 5), c[5]);
          c[6] = fma(a[6u], sub_group_broadcast(b[6], 6), c[6]);
          c[7] = fma(a[6u], sub_group_broadcast(b[6], 7), c[7]);
          c[4] = fma(a[7u], sub_group_broadcast(b[7], 4), c[4]);
          c[5] = fma(a[7u], sub_group_broadcast(b[7], 5), c[5]);
          c[6] = fma(a[7u], sub_group_broadcast(b[7], 6), c[6]);
          c[7] = fma(a[7u], sub_group_broadcast(b[7], 7), c[7]);
          c[8] = fma(a[0u], sub_group_broadcast(b[0], 8), c[8]);
          c[9] = fma(a[0u], sub_group_broadcast(b[0], 9), c[9]);
          c[10] = fma(a[0u], sub_group_broadcast(b[0], 10), c[10]);
          c[11] = fma(a[0u], sub_group_broadcast(b[0], 11), c[11]);
          c[8] = fma(a[1u], sub_group_broadcast(b[1], 8), c[8]);
          c[9] = fma(a[1u], sub_group_broadcast(b[1], 9), c[9]);
          c[10] = fma(a[1u], sub_group_broadcast(b[1], 10), c[10]);
          c[11] = fma(a[1u], sub_group_broadcast(b[1], 11), c[11]);
          c[8] = fma(a[2u], sub_group_broadcast(b[2], 8), c[8]);
          c[9] = fma(a[2u], sub_group_broadcast(b[2], 9), c[9]);
          c[10] = fma(a[2u], sub_group_broadcast(b[2], 10), c[10]);
          c[11] = fma(a[2u], sub_group_broadcast(b[2], 11), c[11]);
          c[8] = fma(a[3u], sub_group_broadcast(b[3], 8), c[8]);
          c[9] = fma(a[3u], sub_group_broadcast(b[3], 9), c[9]);
          c[10] = fma(a[3u], sub_group_broadcast(b[3], 10), c[10]);
          c[11] = fma(a[3u], sub_group_broadcast(b[3], 11), c[11]);
          c[8] = fma(a[4u], sub_group_broadcast(b[4], 8), c[8]);
          c[9] = fma(a[4u], sub_group_broadcast(b[4], 9), c[9]);
          c[10] = fma(a[4u], sub_group_broadcast(b[4], 10), c[10]);
          c[11] = fma(a[4u], sub_group_broadcast(b[4], 11), c[11]);
          c[8] = fma(a[5u], sub_group_broadcast(b[5], 8), c[8]);
          c[9] = fma(a[5u], sub_group_broadcast(b[5], 9), c[9]);
          c[10] = fma(a[5u], sub_group_broadcast(b[5], 10), c[10]);
          c[11] = fma(a[5u], sub_group_broadcast(b[5], 11), c[11]);
          c[8] = fma(a[6u], sub_group_broadcast(b[6], 8), c[8]);
          c[9] = fma(a[6u], sub_group_broadcast(b[6], 9), c[9]);
          c[10] = fma(a[6u], sub_group_broadcast(b[6], 10), c[10]);
          c[11] = fma(a[6u], sub_group_broadcast(b[6], 11), c[11]);
          c[8] = fma(a[7u], sub_group_broadcast(b[7], 8), c[8]);
          c[9] = fma(a[7u], sub_group_broadcast(b[7], 9), c[9]);
          c[10] = fma(a[7u], sub_group_broadcast(b[7], 10), c[10]);
          c[11] = fma(a[7u], sub_group_broadcast(b[7], 11), c[11]);
          c[12] = fma(a[0u], sub_group_broadcast(b[0], 12), c[12]);
          c[13] = fma(a[0u], sub_group_broadcast(b[0], 13), c[13]);
          c[14] = fma(a[0u], sub_group_broadcast(b[0], 14), c[14]);
          c[15] = fma(a[0u], sub_group_broadcast(b[0], 15), c[15]);
          c[12] = fma(a[1u], sub_group_broadcast(b[1], 12), c[12]);
          c[13] = fma(a[1u], sub_group_broadcast(b[1], 13), c[13]);
          c[14] = fma(a[1u], sub_group_broadcast(b[1], 14), c[14]);
          c[15] = fma(a[1u], sub_group_broadcast(b[1], 15), c[15]);
          c[12] = fma(a[2u], sub_group_broadcast(b[2], 12), c[12]);
          c[13] = fma(a[2u], sub_group_broadcast(b[2], 13), c[13]);
          c[14] = fma(a[2u], sub_group_broadcast(b[2], 14), c[14]);
          c[15] = fma(a[2u], sub_group_broadcast(b[2], 15), c[15]);
          c[12] = fma(a[3u], sub_group_broadcast(b[3], 12), c[12]);
          c[13] = fma(a[3u], sub_group_broadcast(b[3], 13), c[13]);
          c[14] = fma(a[3u], sub_group_broadcast(b[3], 14), c[14]);
          c[15] = fma(a[3u], sub_group_broadcast(b[3], 15), c[15]);
          c[12] = fma(a[4u], sub_group_broadcast(b[4], 12), c[12]);
          c[13] = fma(a[4u], sub_group_broadcast(b[4], 13), c[13]);
          c[14] = fma(a[4u], sub_group_broadcast(b[4], 14), c[14]);
          c[15] = fma(a[4u], sub_group_broadcast(b[4], 15), c[15]);
          c[12] = fma(a[5u], sub_group_broadcast(b[5], 12), c[12]);
          c[13] = fma(a[5u], sub_group_broadcast(b[5], 13), c[13]);
          c[14] = fma(a[5u], sub_group_broadcast(b[5], 14), c[14]);
          c[15] = fma(a[5u], sub_group_broadcast(b[5], 15), c[15]);
          c[12] = fma(a[6u], sub_group_broadcast(b[6], 12), c[12]);
          c[13] = fma(a[6u], sub_group_broadcast(b[6], 13), c[13]);
          c[14] = fma(a[6u], sub_group_broadcast(b[6], 14), c[14]);
          c[15] = fma(a[6u], sub_group_broadcast(b[6], 15), c[15]);
          c[12] = fma(a[7u], sub_group_broadcast(b[7], 12), c[12]);
          c[13] = fma(a[7u], sub_group_broadcast(b[7], 13), c[13]);
          c[14] = fma(a[7u], sub_group_broadcast(b[7], 14), c[14]);
          c[15] = fma(a[7u], sub_group_broadcast(b[7], 15), c[15]);
        }
        if (K - KmultipleKb > 0) {
          __attribute__((opencl_unroll_hint(1))) for (short kb = KmultipleKb;
                                                      kb < K; kb += 1) {
            double a[1];
            double b[1];
            if (m < rem1) {
              a[0u] = Ab1[m];
            }
            Ab1 += A_stride1;
            if (m < bs_1) {
              b[0] = Bb2[m];
            }
            Bb2 += B_stride1;
            c[0] = fma(a[0u], sub_group_broadcast(b[0], 0), c[0]);
            c[1] = fma(a[0u], sub_group_broadcast(b[0], 1), c[1]);
            c[2] = fma(a[0u], sub_group_broadcast(b[0], 2), c[2]);
            c[3] = fma(a[0u], sub_group_broadcast(b[0], 3), c[3]);
            c[4] = fma(a[0u], sub_group_broadcast(b[0], 4), c[4]);
            c[5] = fma(a[0u], sub_group_broadcast(b[0], 5), c[5]);
            c[6] = fma(a[0u], sub_group_broadcast(b[0], 6), c[6]);
            c[7] = fma(a[0u], sub_group_broadcast(b[0], 7), c[7]);
            c[8] = fma(a[0u], sub_group_broadcast(b[0], 8), c[8]);
            c[9] = fma(a[0u], sub_group_broadcast(b[0], 9), c[9]);
            c[10] = fma(a[0u], sub_group_broadcast(b[0], 10), c[10]);
            c[11] = fma(a[0u], sub_group_broadcast(b[0], 11), c[11]);
            c[12] = fma(a[0u], sub_group_broadcast(b[0], 12), c[12]);
            c[13] = fma(a[0u], sub_group_broadcast(b[0], 13), c[13]);
            c[14] = fma(a[0u], sub_group_broadcast(b[0], 14), c[14]);
            c[15] = fma(a[0u], sub_group_broadcast(b[0], 15), c[15]);
          }
        }
        __attribute__((opencl_unroll_hint(1))) for (short n = 0;
                                                    n < min(bs_1, (uint)16);
                                                    ++n) {
          if (m < rem1) {
            global double *Cdst = Cb2 + m;
            atomic_fetch_add((global volatile atomic_double *)Cdst,
                             alpha * c[n]);
          }
          Cb2 += (uint)C_stride1;
        }
      }
    }
  }
  __attribute__((opencl_unroll_hint(1))) for (blck = bs_1 * rem; blck < N;
                                              blck += bs) {
    global double *Bb = B + blck;
    global double *Cb = C + C_stride1 * blck;
    uint sg_m = 0;
    uint blocks2 = M / 16u;
    uint rem2 = M % 16u;
    uint blck2;
    __attribute__((opencl_unroll_hint(1))) for (blck2 = 16u * sg_m;
                                                blck2 < 16u * blocks2;
                                                blck2 += 16u) {
      global double *Ab = A + blck2;
      global double *Cb1 = Cb + blck2;
      global double *Ab1 = Ab;
      global double *Bb1 = Bb;
      __attribute__((opencl_unroll_hint(16))) for (short n = 0; n < 16; ++n) {
        c[n] = 0x0p+0;
      }
      uint KmultipleKb = K / 8 * 8;
      __attribute__((opencl_unroll_hint(1))) for (short kb = 0;
                                                  kb < KmultipleKb; kb += 8) {
        double a[8];
        double b[8];
        a[0u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[1u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[2u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[3u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[4u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[5u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[6u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        a[7u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
        Ab1 += A_stride1;
        if (m < bs) {
          b[0] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs) {
          b[1] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs) {
          b[2] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs) {
          b[3] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs) {
          b[4] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs) {
          b[5] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs) {
          b[6] = Bb1[m];
        }
        Bb1 += B_stride1;
        if (m < bs) {
          b[7] = Bb1[m];
        }
        Bb1 += B_stride1;
        c[0] = fma(a[0u], sub_group_broadcast(b[0], 0), c[0]);
        c[1] = fma(a[0u], sub_group_broadcast(b[0], 1), c[1]);
        c[2] = fma(a[0u], sub_group_broadcast(b[0], 2), c[2]);
        c[3] = fma(a[0u], sub_group_broadcast(b[0], 3), c[3]);
        c[0] = fma(a[1u], sub_group_broadcast(b[1], 0), c[0]);
        c[1] = fma(a[1u], sub_group_broadcast(b[1], 1), c[1]);
        c[2] = fma(a[1u], sub_group_broadcast(b[1], 2), c[2]);
        c[3] = fma(a[1u], sub_group_broadcast(b[1], 3), c[3]);
        c[0] = fma(a[2u], sub_group_broadcast(b[2], 0), c[0]);
        c[1] = fma(a[2u], sub_group_broadcast(b[2], 1), c[1]);
        c[2] = fma(a[2u], sub_group_broadcast(b[2], 2), c[2]);
        c[3] = fma(a[2u], sub_group_broadcast(b[2], 3), c[3]);
        c[0] = fma(a[3u], sub_group_broadcast(b[3], 0), c[0]);
        c[1] = fma(a[3u], sub_group_broadcast(b[3], 1), c[1]);
        c[2] = fma(a[3u], sub_group_broadcast(b[3], 2), c[2]);
        c[3] = fma(a[3u], sub_group_broadcast(b[3], 3), c[3]);
        c[0] = fma(a[4u], sub_group_broadcast(b[4], 0), c[0]);
        c[1] = fma(a[4u], sub_group_broadcast(b[4], 1), c[1]);
        c[2] = fma(a[4u], sub_group_broadcast(b[4], 2), c[2]);
        c[3] = fma(a[4u], sub_group_broadcast(b[4], 3), c[3]);
        c[0] = fma(a[5u], sub_group_broadcast(b[5], 0), c[0]);
        c[1] = fma(a[5u], sub_group_broadcast(b[5], 1), c[1]);
        c[2] = fma(a[5u], sub_group_broadcast(b[5], 2), c[2]);
        c[3] = fma(a[5u], sub_group_broadcast(b[5], 3), c[3]);
        c[0] = fma(a[6u], sub_group_broadcast(b[6], 0), c[0]);
        c[1] = fma(a[6u], sub_group_broadcast(b[6], 1), c[1]);
        c[2] = fma(a[6u], sub_group_broadcast(b[6], 2), c[2]);
        c[3] = fma(a[6u], sub_group_broadcast(b[6], 3), c[3]);
        c[0] = fma(a[7u], sub_group_broadcast(b[7], 0), c[0]);
        c[1] = fma(a[7u], sub_group_broadcast(b[7], 1), c[1]);
        c[2] = fma(a[7u], sub_group_broadcast(b[7], 2), c[2]);
        c[3] = fma(a[7u], sub_group_broadcast(b[7], 3), c[3]);
        c[4] = fma(a[0u], sub_group_broadcast(b[0], 4), c[4]);
        c[5] = fma(a[0u], sub_group_broadcast(b[0], 5), c[5]);
        c[6] = fma(a[0u], sub_group_broadcast(b[0], 6), c[6]);
        c[7] = fma(a[0u], sub_group_broadcast(b[0], 7), c[7]);
        c[4] = fma(a[1u], sub_group_broadcast(b[1], 4), c[4]);
        c[5] = fma(a[1u], sub_group_broadcast(b[1], 5), c[5]);
        c[6] = fma(a[1u], sub_group_broadcast(b[1], 6), c[6]);
        c[7] = fma(a[1u], sub_group_broadcast(b[1], 7), c[7]);
        c[4] = fma(a[2u], sub_group_broadcast(b[2], 4), c[4]);
        c[5] = fma(a[2u], sub_group_broadcast(b[2], 5), c[5]);
        c[6] = fma(a[2u], sub_group_broadcast(b[2], 6), c[6]);
        c[7] = fma(a[2u], sub_group_broadcast(b[2], 7), c[7]);
        c[4] = fma(a[3u], sub_group_broadcast(b[3], 4), c[4]);
        c[5] = fma(a[3u], sub_group_broadcast(b[3], 5), c[5]);
        c[6] = fma(a[3u], sub_group_broadcast(b[3], 6), c[6]);
        c[7] = fma(a[3u], sub_group_broadcast(b[3], 7), c[7]);
        c[4] = fma(a[4u], sub_group_broadcast(b[4], 4), c[4]);
        c[5] = fma(a[4u], sub_group_broadcast(b[4], 5), c[5]);
        c[6] = fma(a[4u], sub_group_broadcast(b[4], 6), c[6]);
        c[7] = fma(a[4u], sub_group_broadcast(b[4], 7), c[7]);
        c[4] = fma(a[5u], sub_group_broadcast(b[5], 4), c[4]);
        c[5] = fma(a[5u], sub_group_broadcast(b[5], 5), c[5]);
        c[6] = fma(a[5u], sub_group_broadcast(b[5], 6), c[6]);
        c[7] = fma(a[5u], sub_group_broadcast(b[5], 7), c[7]);
        c[4] = fma(a[6u], sub_group_broadcast(b[6], 4), c[4]);
        c[5] = fma(a[6u], sub_group_broadcast(b[6], 5), c[5]);
        c[6] = fma(a[6u], sub_group_broadcast(b[6], 6), c[6]);
        c[7] = fma(a[6u], sub_group_broadcast(b[6], 7), c[7]);
        c[4] = fma(a[7u], sub_group_broadcast(b[7], 4), c[4]);
        c[5] = fma(a[7u], sub_group_broadcast(b[7], 5), c[5]);
        c[6] = fma(a[7u], sub_group_broadcast(b[7], 6), c[6]);
        c[7] = fma(a[7u], sub_group_broadcast(b[7], 7), c[7]);
        c[8] = fma(a[0u], sub_group_broadcast(b[0], 8), c[8]);
        c[9] = fma(a[0u], sub_group_broadcast(b[0], 9), c[9]);
        c[10] = fma(a[0u], sub_group_broadcast(b[0], 10), c[10]);
        c[11] = fma(a[0u], sub_group_broadcast(b[0], 11), c[11]);
        c[8] = fma(a[1u], sub_group_broadcast(b[1], 8), c[8]);
        c[9] = fma(a[1u], sub_group_broadcast(b[1], 9), c[9]);
        c[10] = fma(a[1u], sub_group_broadcast(b[1], 10), c[10]);
        c[11] = fma(a[1u], sub_group_broadcast(b[1], 11), c[11]);
        c[8] = fma(a[2u], sub_group_broadcast(b[2], 8), c[8]);
        c[9] = fma(a[2u], sub_group_broadcast(b[2], 9), c[9]);
        c[10] = fma(a[2u], sub_group_broadcast(b[2], 10), c[10]);
        c[11] = fma(a[2u], sub_group_broadcast(b[2], 11), c[11]);
        c[8] = fma(a[3u], sub_group_broadcast(b[3], 8), c[8]);
        c[9] = fma(a[3u], sub_group_broadcast(b[3], 9), c[9]);
        c[10] = fma(a[3u], sub_group_broadcast(b[3], 10), c[10]);
        c[11] = fma(a[3u], sub_group_broadcast(b[3], 11), c[11]);
        c[8] = fma(a[4u], sub_group_broadcast(b[4], 8), c[8]);
        c[9] = fma(a[4u], sub_group_broadcast(b[4], 9), c[9]);
        c[10] = fma(a[4u], sub_group_broadcast(b[4], 10), c[10]);
        c[11] = fma(a[4u], sub_group_broadcast(b[4], 11), c[11]);
        c[8] = fma(a[5u], sub_group_broadcast(b[5], 8), c[8]);
        c[9] = fma(a[5u], sub_group_broadcast(b[5], 9), c[9]);
        c[10] = fma(a[5u], sub_group_broadcast(b[5], 10), c[10]);
        c[11] = fma(a[5u], sub_group_broadcast(b[5], 11), c[11]);
        c[8] = fma(a[6u], sub_group_broadcast(b[6], 8), c[8]);
        c[9] = fma(a[6u], sub_group_broadcast(b[6], 9), c[9]);
        c[10] = fma(a[6u], sub_group_broadcast(b[6], 10), c[10]);
        c[11] = fma(a[6u], sub_group_broadcast(b[6], 11), c[11]);
        c[8] = fma(a[7u], sub_group_broadcast(b[7], 8), c[8]);
        c[9] = fma(a[7u], sub_group_broadcast(b[7], 9), c[9]);
        c[10] = fma(a[7u], sub_group_broadcast(b[7], 10), c[10]);
        c[11] = fma(a[7u], sub_group_broadcast(b[7], 11), c[11]);
        c[12] = fma(a[0u], sub_group_broadcast(b[0], 12), c[12]);
        c[13] = fma(a[0u], sub_group_broadcast(b[0], 13), c[13]);
        c[14] = fma(a[0u], sub_group_broadcast(b[0], 14), c[14]);
        c[15] = fma(a[0u], sub_group_broadcast(b[0], 15), c[15]);
        c[12] = fma(a[1u], sub_group_broadcast(b[1], 12), c[12]);
        c[13] = fma(a[1u], sub_group_broadcast(b[1], 13), c[13]);
        c[14] = fma(a[1u], sub_group_broadcast(b[1], 14), c[14]);
        c[15] = fma(a[1u], sub_group_broadcast(b[1], 15), c[15]);
        c[12] = fma(a[2u], sub_group_broadcast(b[2], 12), c[12]);
        c[13] = fma(a[2u], sub_group_broadcast(b[2], 13), c[13]);
        c[14] = fma(a[2u], sub_group_broadcast(b[2], 14), c[14]);
        c[15] = fma(a[2u], sub_group_broadcast(b[2], 15), c[15]);
        c[12] = fma(a[3u], sub_group_broadcast(b[3], 12), c[12]);
        c[13] = fma(a[3u], sub_group_broadcast(b[3], 13), c[13]);
        c[14] = fma(a[3u], sub_group_broadcast(b[3], 14), c[14]);
        c[15] = fma(a[3u], sub_group_broadcast(b[3], 15), c[15]);
        c[12] = fma(a[4u], sub_group_broadcast(b[4], 12), c[12]);
        c[13] = fma(a[4u], sub_group_broadcast(b[4], 13), c[13]);
        c[14] = fma(a[4u], sub_group_broadcast(b[4], 14), c[14]);
        c[15] = fma(a[4u], sub_group_broadcast(b[4], 15), c[15]);
        c[12] = fma(a[5u], sub_group_broadcast(b[5], 12), c[12]);
        c[13] = fma(a[5u], sub_group_broadcast(b[5], 13), c[13]);
        c[14] = fma(a[5u], sub_group_broadcast(b[5], 14), c[14]);
        c[15] = fma(a[5u], sub_group_broadcast(b[5], 15), c[15]);
        c[12] = fma(a[6u], sub_group_broadcast(b[6], 12), c[12]);
        c[13] = fma(a[6u], sub_group_broadcast(b[6], 13), c[13]);
        c[14] = fma(a[6u], sub_group_broadcast(b[6], 14), c[14]);
        c[15] = fma(a[6u], sub_group_broadcast(b[6], 15), c[15]);
        c[12] = fma(a[7u], sub_group_broadcast(b[7], 12), c[12]);
        c[13] = fma(a[7u], sub_group_broadcast(b[7], 13), c[13]);
        c[14] = fma(a[7u], sub_group_broadcast(b[7], 14), c[14]);
        c[15] = fma(a[7u], sub_group_broadcast(b[7], 15), c[15]);
      }
      if (K - KmultipleKb > 0) {
        __attribute__((opencl_unroll_hint(1))) for (short kb = KmultipleKb;
                                                    kb < K; kb += 1) {
          double a[1];
          double b[1];
          a[0u] = as_double(intel_sub_group_block_read_ul((global ulong *)Ab1));
          Ab1 += A_stride1;
          if (m < bs) {
            b[0] = Bb1[m];
          }
          Bb1 += B_stride1;
          c[0] = fma(a[0u], sub_group_broadcast(b[0], 0), c[0]);
          c[1] = fma(a[0u], sub_group_broadcast(b[0], 1), c[1]);
          c[2] = fma(a[0u], sub_group_broadcast(b[0], 2), c[2]);
          c[3] = fma(a[0u], sub_group_broadcast(b[0], 3), c[3]);
          c[4] = fma(a[0u], sub_group_broadcast(b[0], 4), c[4]);
          c[5] = fma(a[0u], sub_group_broadcast(b[0], 5), c[5]);
          c[6] = fma(a[0u], sub_group_broadcast(b[0], 6), c[6]);
          c[7] = fma(a[0u], sub_group_broadcast(b[0], 7), c[7]);
          c[8] = fma(a[0u], sub_group_broadcast(b[0], 8), c[8]);
          c[9] = fma(a[0u], sub_group_broadcast(b[0], 9), c[9]);
          c[10] = fma(a[0u], sub_group_broadcast(b[0], 10), c[10]);
          c[11] = fma(a[0u], sub_group_broadcast(b[0], 11), c[11]);
          c[12] = fma(a[0u], sub_group_broadcast(b[0], 12), c[12]);
          c[13] = fma(a[0u], sub_group_broadcast(b[0], 13), c[13]);
          c[14] = fma(a[0u], sub_group_broadcast(b[0], 14), c[14]);
          c[15] = fma(a[0u], sub_group_broadcast(b[0], 15), c[15]);
        }
      }
      __attribute__((opencl_unroll_hint(1))) for (short n = 0;
                                                  n < min(bs, (uint)16); ++n) {
        global double *Cdst = Cb1 + m;
        atomic_fetch_add((global volatile atomic_double *)Cdst, alpha * c[n]);
        Cb1 += (uint)C_stride1;
      }
    }
    if (rem2 > 0) {
      blck2 = blocks2 * 16u;
      if (sg_m == 0u) {
        global double *Ab = A + blck2;
        global double *Cb2 = Cb + blck2;
        global double *Ab1 = Ab;
        global double *Bb2 = Bb;
        __attribute__((opencl_unroll_hint(16))) for (short n = 0; n < 16; ++n) {
          c[n] = 0x0p+0;
        }
        uint KmultipleKb = K / 8 * 8;
        __attribute__((opencl_unroll_hint(1))) for (short kb = 0;
                                                    kb < KmultipleKb; kb += 8) {
          double a[8];
          double b[8];
          if (m < rem2) {
            a[0u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem2) {
            a[1u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem2) {
            a[2u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem2) {
            a[3u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem2) {
            a[4u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem2) {
            a[5u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem2) {
            a[6u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < rem2) {
            a[7u] = Ab1[m];
          }
          Ab1 += A_stride1;
          if (m < bs) {
            b[0] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs) {
            b[1] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs) {
            b[2] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs) {
            b[3] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs) {
            b[4] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs) {
            b[5] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs) {
            b[6] = Bb2[m];
          }
          Bb2 += B_stride1;
          if (m < bs) {
            b[7] = Bb2[m];
          }
          Bb2 += B_stride1;
          c[0] = fma(a[0u], sub_group_broadcast(b[0], 0), c[0]);
          c[1] = fma(a[0u], sub_group_broadcast(b[0], 1), c[1]);
          c[2] = fma(a[0u], sub_group_broadcast(b[0], 2), c[2]);
          c[3] = fma(a[0u], sub_group_broadcast(b[0], 3), c[3]);
          c[0] = fma(a[1u], sub_group_broadcast(b[1], 0), c[0]);
          c[1] = fma(a[1u], sub_group_broadcast(b[1], 1), c[1]);
          c[2] = fma(a[1u], sub_group_broadcast(b[1], 2), c[2]);
          c[3] = fma(a[1u], sub_group_broadcast(b[1], 3), c[3]);
          c[0] = fma(a[2u], sub_group_broadcast(b[2], 0), c[0]);
          c[1] = fma(a[2u], sub_group_broadcast(b[2], 1), c[1]);
          c[2] = fma(a[2u], sub_group_broadcast(b[2], 2), c[2]);
          c[3] = fma(a[2u], sub_group_broadcast(b[2], 3), c[3]);
          c[0] = fma(a[3u], sub_group_broadcast(b[3], 0), c[0]);
          c[1] = fma(a[3u], sub_group_broadcast(b[3], 1), c[1]);
          c[2] = fma(a[3u], sub_group_broadcast(b[3], 2), c[2]);
          c[3] = fma(a[3u], sub_group_broadcast(b[3], 3), c[3]);
          c[0] = fma(a[4u], sub_group_broadcast(b[4], 0), c[0]);
          c[1] = fma(a[4u], sub_group_broadcast(b[4], 1), c[1]);
          c[2] = fma(a[4u], sub_group_broadcast(b[4], 2), c[2]);
          c[3] = fma(a[4u], sub_group_broadcast(b[4], 3), c[3]);
          c[0] = fma(a[5u], sub_group_broadcast(b[5], 0), c[0]);
          c[1] = fma(a[5u], sub_group_broadcast(b[5], 1), c[1]);
          c[2] = fma(a[5u], sub_group_broadcast(b[5], 2), c[2]);
          c[3] = fma(a[5u], sub_group_broadcast(b[5], 3), c[3]);
          c[0] = fma(a[6u], sub_group_broadcast(b[6], 0), c[0]);
          c[1] = fma(a[6u], sub_group_broadcast(b[6], 1), c[1]);
          c[2] = fma(a[6u], sub_group_broadcast(b[6], 2), c[2]);
          c[3] = fma(a[6u], sub_group_broadcast(b[6], 3), c[3]);
          c[0] = fma(a[7u], sub_group_broadcast(b[7], 0), c[0]);
          c[1] = fma(a[7u], sub_group_broadcast(b[7], 1), c[1]);
          c[2] = fma(a[7u], sub_group_broadcast(b[7], 2), c[2]);
          c[3] = fma(a[7u], sub_group_broadcast(b[7], 3), c[3]);
          c[4] = fma(a[0u], sub_group_broadcast(b[0], 4), c[4]);
          c[5] = fma(a[0u], sub_group_broadcast(b[0], 5), c[5]);
          c[6] = fma(a[0u], sub_group_broadcast(b[0], 6), c[6]);
          c[7] = fma(a[0u], sub_group_broadcast(b[0], 7), c[7]);
          c[4] = fma(a[1u], sub_group_broadcast(b[1], 4), c[4]);
          c[5] = fma(a[1u], sub_group_broadcast(b[1], 5), c[5]);
          c[6] = fma(a[1u], sub_group_broadcast(b[1], 6), c[6]);
          c[7] = fma(a[1u], sub_group_broadcast(b[1], 7), c[7]);
          c[4] = fma(a[2u], sub_group_broadcast(b[2], 4), c[4]);
          c[5] = fma(a[2u], sub_group_broadcast(b[2], 5), c[5]);
          c[6] = fma(a[2u], sub_group_broadcast(b[2], 6), c[6]);
          c[7] = fma(a[2u], sub_group_broadcast(b[2], 7), c[7]);
          c[4] = fma(a[3u], sub_group_broadcast(b[3], 4), c[4]);
          c[5] = fma(a[3u], sub_group_broadcast(b[3], 5), c[5]);
          c[6] = fma(a[3u], sub_group_broadcast(b[3], 6), c[6]);
          c[7] = fma(a[3u], sub_group_broadcast(b[3], 7), c[7]);
          c[4] = fma(a[4u], sub_group_broadcast(b[4], 4), c[4]);
          c[5] = fma(a[4u], sub_group_broadcast(b[4], 5), c[5]);
          c[6] = fma(a[4u], sub_group_broadcast(b[4], 6), c[6]);
          c[7] = fma(a[4u], sub_group_broadcast(b[4], 7), c[7]);
          c[4] = fma(a[5u], sub_group_broadcast(b[5], 4), c[4]);
          c[5] = fma(a[5u], sub_group_broadcast(b[5], 5), c[5]);
          c[6] = fma(a[5u], sub_group_broadcast(b[5], 6), c[6]);
          c[7] = fma(a[5u], sub_group_broadcast(b[5], 7), c[7]);
          c[4] = fma(a[6u], sub_group_broadcast(b[6], 4), c[4]);
          c[5] = fma(a[6u], sub_group_broadcast(b[6], 5), c[5]);
          c[6] = fma(a[6u], sub_group_broadcast(b[6], 6), c[6]);
          c[7] = fma(a[6u], sub_group_broadcast(b[6], 7), c[7]);
          c[4] = fma(a[7u], sub_group_broadcast(b[7], 4), c[4]);
          c[5] = fma(a[7u], sub_group_broadcast(b[7], 5), c[5]);
          c[6] = fma(a[7u], sub_group_broadcast(b[7], 6), c[6]);
          c[7] = fma(a[7u], sub_group_broadcast(b[7], 7), c[7]);
          c[8] = fma(a[0u], sub_group_broadcast(b[0], 8), c[8]);
          c[9] = fma(a[0u], sub_group_broadcast(b[0], 9), c[9]);
          c[10] = fma(a[0u], sub_group_broadcast(b[0], 10), c[10]);
          c[11] = fma(a[0u], sub_group_broadcast(b[0], 11), c[11]);
          c[8] = fma(a[1u], sub_group_broadcast(b[1], 8), c[8]);
          c[9] = fma(a[1u], sub_group_broadcast(b[1], 9), c[9]);
          c[10] = fma(a[1u], sub_group_broadcast(b[1], 10), c[10]);
          c[11] = fma(a[1u], sub_group_broadcast(b[1], 11), c[11]);
          c[8] = fma(a[2u], sub_group_broadcast(b[2], 8), c[8]);
          c[9] = fma(a[2u], sub_group_broadcast(b[2], 9), c[9]);
          c[10] = fma(a[2u], sub_group_broadcast(b[2], 10), c[10]);
          c[11] = fma(a[2u], sub_group_broadcast(b[2], 11), c[11]);
          c[8] = fma(a[3u], sub_group_broadcast(b[3], 8), c[8]);
          c[9] = fma(a[3u], sub_group_broadcast(b[3], 9), c[9]);
          c[10] = fma(a[3u], sub_group_broadcast(b[3], 10), c[10]);
          c[11] = fma(a[3u], sub_group_broadcast(b[3], 11), c[11]);
          c[8] = fma(a[4u], sub_group_broadcast(b[4], 8), c[8]);
          c[9] = fma(a[4u], sub_group_broadcast(b[4], 9), c[9]);
          c[10] = fma(a[4u], sub_group_broadcast(b[4], 10), c[10]);
          c[11] = fma(a[4u], sub_group_broadcast(b[4], 11), c[11]);
          c[8] = fma(a[5u], sub_group_broadcast(b[5], 8), c[8]);
          c[9] = fma(a[5u], sub_group_broadcast(b[5], 9), c[9]);
          c[10] = fma(a[5u], sub_group_broadcast(b[5], 10), c[10]);
          c[11] = fma(a[5u], sub_group_broadcast(b[5], 11), c[11]);
          c[8] = fma(a[6u], sub_group_broadcast(b[6], 8), c[8]);
          c[9] = fma(a[6u], sub_group_broadcast(b[6], 9), c[9]);
          c[10] = fma(a[6u], sub_group_broadcast(b[6], 10), c[10]);
          c[11] = fma(a[6u], sub_group_broadcast(b[6], 11), c[11]);
          c[8] = fma(a[7u], sub_group_broadcast(b[7], 8), c[8]);
          c[9] = fma(a[7u], sub_group_broadcast(b[7], 9), c[9]);
          c[10] = fma(a[7u], sub_group_broadcast(b[7], 10), c[10]);
          c[11] = fma(a[7u], sub_group_broadcast(b[7], 11), c[11]);
          c[12] = fma(a[0u], sub_group_broadcast(b[0], 12), c[12]);
          c[13] = fma(a[0u], sub_group_broadcast(b[0], 13), c[13]);
          c[14] = fma(a[0u], sub_group_broadcast(b[0], 14), c[14]);
          c[15] = fma(a[0u], sub_group_broadcast(b[0], 15), c[15]);
          c[12] = fma(a[1u], sub_group_broadcast(b[1], 12), c[12]);
          c[13] = fma(a[1u], sub_group_broadcast(b[1], 13), c[13]);
          c[14] = fma(a[1u], sub_group_broadcast(b[1], 14), c[14]);
          c[15] = fma(a[1u], sub_group_broadcast(b[1], 15), c[15]);
          c[12] = fma(a[2u], sub_group_broadcast(b[2], 12), c[12]);
          c[13] = fma(a[2u], sub_group_broadcast(b[2], 13), c[13]);
          c[14] = fma(a[2u], sub_group_broadcast(b[2], 14), c[14]);
          c[15] = fma(a[2u], sub_group_broadcast(b[2], 15), c[15]);
          c[12] = fma(a[3u], sub_group_broadcast(b[3], 12), c[12]);
          c[13] = fma(a[3u], sub_group_broadcast(b[3], 13), c[13]);
          c[14] = fma(a[3u], sub_group_broadcast(b[3], 14), c[14]);
          c[15] = fma(a[3u], sub_group_broadcast(b[3], 15), c[15]);
          c[12] = fma(a[4u], sub_group_broadcast(b[4], 12), c[12]);
          c[13] = fma(a[4u], sub_group_broadcast(b[4], 13), c[13]);
          c[14] = fma(a[4u], sub_group_broadcast(b[4], 14), c[14]);
          c[15] = fma(a[4u], sub_group_broadcast(b[4], 15), c[15]);
          c[12] = fma(a[5u], sub_group_broadcast(b[5], 12), c[12]);
          c[13] = fma(a[5u], sub_group_broadcast(b[5], 13), c[13]);
          c[14] = fma(a[5u], sub_group_broadcast(b[5], 14), c[14]);
          c[15] = fma(a[5u], sub_group_broadcast(b[5], 15), c[15]);
          c[12] = fma(a[6u], sub_group_broadcast(b[6], 12), c[12]);
          c[13] = fma(a[6u], sub_group_broadcast(b[6], 13), c[13]);
          c[14] = fma(a[6u], sub_group_broadcast(b[6], 14), c[14]);
          c[15] = fma(a[6u], sub_group_broadcast(b[6], 15), c[15]);
          c[12] = fma(a[7u], sub_group_broadcast(b[7], 12), c[12]);
          c[13] = fma(a[7u], sub_group_broadcast(b[7], 13), c[13]);
          c[14] = fma(a[7u], sub_group_broadcast(b[7], 14), c[14]);
          c[15] = fma(a[7u], sub_group_broadcast(b[7], 15), c[15]);
        }
        if (K - KmultipleKb > 0) {
          __attribute__((opencl_unroll_hint(1))) for (short kb = KmultipleKb;
                                                      kb < K; kb += 1) {
            double a[1];
            double b[1];
            if (m < rem2) {
              a[0u] = Ab1[m];
            }
            Ab1 += A_stride1;
            if (m < bs) {
              b[0] = Bb2[m];
            }
            Bb2 += B_stride1;
            c[0] = fma(a[0u], sub_group_broadcast(b[0], 0), c[0]);
            c[1] = fma(a[0u], sub_group_broadcast(b[0], 1), c[1]);
            c[2] = fma(a[0u], sub_group_broadcast(b[0], 2), c[2]);
            c[3] = fma(a[0u], sub_group_broadcast(b[0], 3), c[3]);
            c[4] = fma(a[0u], sub_group_broadcast(b[0], 4), c[4]);
            c[5] = fma(a[0u], sub_group_broadcast(b[0], 5), c[5]);
            c[6] = fma(a[0u], sub_group_broadcast(b[0], 6), c[6]);
            c[7] = fma(a[0u], sub_group_broadcast(b[0], 7), c[7]);
            c[8] = fma(a[0u], sub_group_broadcast(b[0], 8), c[8]);
            c[9] = fma(a[0u], sub_group_broadcast(b[0], 9), c[9]);
            c[10] = fma(a[0u], sub_group_broadcast(b[0], 10), c[10]);
            c[11] = fma(a[0u], sub_group_broadcast(b[0], 11), c[11]);
            c[12] = fma(a[0u], sub_group_broadcast(b[0], 12), c[12]);
            c[13] = fma(a[0u], sub_group_broadcast(b[0], 13), c[13]);
            c[14] = fma(a[0u], sub_group_broadcast(b[0], 14), c[14]);
            c[15] = fma(a[0u], sub_group_broadcast(b[0], 15), c[15]);
          }
        }
        __attribute__((opencl_unroll_hint(1))) for (short n = 0;
                                                    n < min(bs, (uint)16);
                                                    ++n) {
          if (m < rem2) {
            global double *Cdst = Cb2 + m;
            atomic_fetch_add((global volatile atomic_double *)Cdst,
                             alpha * c[n]);
          }
          Cb2 += (uint)C_stride1;
        }
      }
    }
  }
}
kernel __attribute__((reqd_work_group_size(16, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16))) void
dbm_multiply(double alpha, int itask, global int *tasks, uint tasks_shape1,
             global double *A, uint A_shape0, global double *B, uint B_shape0,
             global double *C, uint C_shape0) {
  uint gid = get_global_id(2);
  uint itask_idx = (uint)itask;
  uint tid = itask_idx + gid;
  int iM = *(tasks + 0 * 1 + tid * 6);
  int iN = *(tasks + 1 * 1 + tid * 6);
  int iK = *(tasks + 2 * 1 + tid * 6);
  int ioffset_a = *(tasks + 3 * 1 + tid * 6);
  int ioffset_b = *(tasks + 4 * 1 + tid * 6);
  int ioffset_c = *(tasks + 5 * 1 + tid * 6);
  uint M = (uint)iM;
  uint N = (uint)iN;
  uint K = (uint)iK;
  uint offset_a = (uint)ioffset_a;
  uint offset_b = (uint)ioffset_b;
  uint offset_c = (uint)ioffset_c;
  uint MK = M * K;
  uint KN = K * N;
  uint MN = M * N;
  global double *av = A + offset_a * 1;
  uint av_shape0 = MK;
  global double *bv = B + offset_b * 1;
  uint bv_shape0 = KN;
  global double *cv = C + offset_c * 1;
  uint cv_shape0 = MN;
  global double *a = av;
  uint a_shape0 = M;
  uint a_shape1 = K;
  uint a_stride1 = 1 * M;
  global double *b = bv;
  uint b_shape0 = N;
  uint b_shape1 = K;
  uint b_stride1 = 1 * N;
  global double *c = cv;
  uint c_shape0 = M;
  uint c_shape1 = N;
  uint c_stride1 = 1 * M;
  gemm_atomic_f64f64f64f64f64_An_Bt_Md_Nd_Kd_Astride1_d_Bstride1_d_Cstride1_d_alphad_beta3ff0000000000000(
      c_shape0, c_shape1, a_shape1, alpha, a, 1, a_stride1, b, 1, b_stride1,
      0x1p+0, c, 1, c_stride1);
}
