/*
 *  Copyright 2026 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/row.h"
#include "libyuv/scale_row.h"

#if !defined(LIBYUV_DISABLE_VSX) && defined(__VSX__) && defined(_ARCH_PWR8) && \
    defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__

#include <altivec.h>
#include <string.h>

#undef vector
#undef pixel
#undef bool

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

typedef __vector unsigned char v8u;
typedef __vector unsigned short v16u;
typedef __vector unsigned int v32u;
typedef __vector unsigned long long v64u;

// Constant byte shuffle of two v8u. The indices count elements in memory
// order on both byte orders.
#if defined(__clang__) || __GNUC__ >= 12
#define SHUFFLE_VSX(a, b, ...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
#define SHUFFLE_VSX(a, b, ...) __builtin_shuffle(a, b, (v8u){__VA_ARGS__})
#endif

static __inline v8u Load16_VSX(const uint8_t* p) {
  return vec_xl(0, p);
}

static __inline void Store16_VSX(v8u v, uint8_t* p) {
  vec_xst(v, 0, p);
}

static __inline unsigned long long Load8u_VSX(const uint8_t* p) {
  unsigned long long v;
  memcpy(&v, p, 8);
  return v;
}

static __inline uint32_t Load4u_VSX(const uint8_t* p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}

// Average 2x2 boxes of ARGB pixels, rounded. s and t hold the same 4 pixels
// (2 output pixels) from two rows.
static __inline v32u Box2_VSX(v8u s, v8u t, v8u one, v32u two, int hi) {
  v8u q = hi ? SHUFFLE_VSX(s, t, 8, 12, 24, 28, 9, 13, 25, 29, 10, 14, 26, 30,
                           11, 15, 27, 31)
             : SHUFFLE_VSX(s, t, 0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7,
                           19, 23);
  return vec_msum(q, one, two);
}

// Returns 4 ARGB pixels averaged from the 2x2 boxes of s0, s1 (row 0) and
// t0, t1 (row 1).
static __inline v8u BoxARGB_VSX(v8u s0, v8u s1, v8u t0, v8u t1) {
  v8u one = vec_splats((unsigned char)1);
  v32u two = vec_splats(2u);
  v16u k2 = vec_splats((unsigned short)2);
  v16u lo =
      vec_pack(Box2_VSX(s0, t0, one, two, 0), Box2_VSX(s0, t0, one, two, 1));
  v16u hi =
      vec_pack(Box2_VSX(s1, t1, one, two, 0), Box2_VSX(s1, t1, one, two, 1));
  return (v8u)vec_pack(vec_sr(lo, k2), vec_sr(hi, k2));
}

#define EVEN_ARGB_VSX 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
#define ODD_ARGB_VSX 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

void ScaleARGBRowDown2_VSX(const uint8_t* src_argb,
                           ptrdiff_t src_stride,
                           uint8_t* dst_argb,
                           int dst_width) {
  (void)src_stride;
  do {
    v8u a0 = Load16_VSX(src_argb);
    v8u a1 = Load16_VSX(src_argb + 16);
    v8u a2 = Load16_VSX(src_argb + 32);
    v8u a3 = Load16_VSX(src_argb + 48);
    Store16_VSX(SHUFFLE_VSX(a0, a1, ODD_ARGB_VSX), dst_argb);
    Store16_VSX(SHUFFLE_VSX(a2, a3, ODD_ARGB_VSX), dst_argb + 16);
    src_argb += 64;
    dst_argb += 32;
    dst_width -= 8;
  } while (dst_width > 0);
}

void ScaleARGBRowDown2Linear_VSX(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_argb,
                                 int dst_width) {
  (void)src_stride;
  do {
    v8u a0 = Load16_VSX(src_argb);
    v8u a1 = Load16_VSX(src_argb + 16);
    v8u a2 = Load16_VSX(src_argb + 32);
    v8u a3 = Load16_VSX(src_argb + 48);
    Store16_VSX(vec_avg(SHUFFLE_VSX(a0, a1, EVEN_ARGB_VSX),
                        SHUFFLE_VSX(a0, a1, ODD_ARGB_VSX)),
                dst_argb);
    Store16_VSX(vec_avg(SHUFFLE_VSX(a2, a3, EVEN_ARGB_VSX),
                        SHUFFLE_VSX(a2, a3, ODD_ARGB_VSX)),
                dst_argb + 16);
    src_argb += 64;
    dst_argb += 32;
    dst_width -= 8;
  } while (dst_width > 0);
}

void ScaleARGBRowDown2Box_VSX(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              uint8_t* dst_argb,
                              int dst_width) {
  const uint8_t* t = src_argb + src_stride;
  do {
    v8u s0 = Load16_VSX(src_argb);
    v8u s1 = Load16_VSX(src_argb + 16);
    v8u s2 = Load16_VSX(src_argb + 32);
    v8u s3 = Load16_VSX(src_argb + 48);
    v8u t0 = Load16_VSX(t);
    v8u t1 = Load16_VSX(t + 16);
    v8u t2 = Load16_VSX(t + 32);
    v8u t3 = Load16_VSX(t + 48);
    Store16_VSX(BoxARGB_VSX(s0, s1, t0, t1), dst_argb);
    Store16_VSX(BoxARGB_VSX(s2, s3, t2, t3), dst_argb + 16);
    src_argb += 64;
    t += 64;
    dst_argb += 32;
    dst_width -= 8;
  } while (dst_width > 0);
}

void ScaleARGBRowDownEven_VSX(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              int src_stepx,
                              uint8_t* dst_argb,
                              int dst_width) {
  const ptrdiff_t step = (ptrdiff_t)src_stepx * 4;
  (void)src_stride;
  do {
    v32u v = {Load4u_VSX(src_argb), Load4u_VSX(src_argb + step),
              Load4u_VSX(src_argb + step * 2), Load4u_VSX(src_argb + step * 3)};
    Store16_VSX((v8u)v, dst_argb);
    src_argb += step * 4;
    dst_argb += 16;
    dst_width -= 4;
  } while (dst_width > 0);
}

void ScaleARGBRowDownEvenBox_VSX(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 int src_stepx,
                                 uint8_t* dst_argb,
                                 int dst_width) {
  const ptrdiff_t step = (ptrdiff_t)src_stepx * 4;
  do {
    const uint8_t* t = src_argb + src_stride;
    v64u s0 = {Load8u_VSX(src_argb), Load8u_VSX(src_argb + step)};
    v64u s1 = {Load8u_VSX(src_argb + step * 2),
               Load8u_VSX(src_argb + step * 3)};
    v64u t0 = {Load8u_VSX(t), Load8u_VSX(t + step)};
    v64u t1 = {Load8u_VSX(t + step * 2), Load8u_VSX(t + step * 3)};
    Store16_VSX(BoxARGB_VSX((v8u)s0, (v8u)s1, (v8u)t0, (v8u)t1), dst_argb);
    src_argb += step * 4;
    dst_argb += 16;
    dst_width -= 4;
  } while (dst_width > 0);
}

void ScaleARGBCols_VSX(uint8_t* dst_argb,
                       const uint8_t* src_argb,
                       int dst_width,
                       int x,
                       int dx) {
  uint32_t ux = (uint32_t)x;
  const uint32_t udx = (uint32_t)dx;
  do {
    uint32_t p0 = Load4u_VSX(src_argb + ((int)ux >> 16) * 4);
    uint32_t p1 = Load4u_VSX(src_argb + ((int)(ux + udx) >> 16) * 4);
    uint32_t p2 = Load4u_VSX(src_argb + ((int)(ux + udx * 2) >> 16) * 4);
    uint32_t p3 = Load4u_VSX(src_argb + ((int)(ux + udx * 3) >> 16) * 4);
    v32u v = {p0, p1, p2, p3};
    Store16_VSX((v8u)v, dst_argb);
    ux += udx * 4;
    dst_argb += 16;
    dst_width -= 4;
  } while (dst_width > 0);
}

#define DUP_ARGB_LO_VSX 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7
#define DUP_ARGB_HI_VSX \
  8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15

void ScaleARGBColsUp2_VSX(uint8_t* dst_argb,
                          const uint8_t* src_argb,
                          int dst_width,
                          int x,
                          int dx) {
  (void)x;
  (void)dx;
  do {
    v8u s0 = Load16_VSX(src_argb);
    v8u s1 = Load16_VSX(src_argb + 16);
    Store16_VSX(SHUFFLE_VSX(s0, s0, DUP_ARGB_LO_VSX), dst_argb);
    Store16_VSX(SHUFFLE_VSX(s0, s0, DUP_ARGB_HI_VSX), dst_argb + 16);
    Store16_VSX(SHUFFLE_VSX(s1, s1, DUP_ARGB_LO_VSX), dst_argb + 32);
    Store16_VSX(SHUFFLE_VSX(s1, s1, DUP_ARGB_HI_VSX), dst_argb + 48);
    src_argb += 32;
    dst_argb += 64;
    dst_width -= 16;
  } while (dst_width > 0);
}

// Bilinear ARGB column filtering with 7 bit fractions, matching
// ScaleARGBFilterCols_C: (a * (127 - f) + b * f) >> 7. Only the low 16 bits
// of x give the fraction, so x is also kept in 16-bit lanes, two per pixel to
// match the even and odd bytes of each pixel.
void ScaleARGBFilterCols_VSX(uint8_t* dst_argb,
                             const uint8_t* src_argb,
                             int dst_width,
                             int x,
                             int dx) {
  uint32_t ux = (uint32_t)x;
  const uint32_t udx = (uint32_t)dx;
  unsigned short x0 = (unsigned short)ux;
  unsigned short x1 = (unsigned short)(ux + udx);
  unsigned short x2 = (unsigned short)(ux + udx * 2);
  unsigned short x3 = (unsigned short)(ux + udx * 3);
  v16u xs = {x0, x0, x1, x1, x2, x2, x3, x3};
  const v16u dx4 = vec_splats((unsigned short)(udx * 4));
  const v16u k9 = vec_splats((unsigned short)9);
  const v16u k514 = vec_splats((unsigned short)514);
  const v16u zero = vec_splats((unsigned short)0);
  const v8u kfe = vec_splats((unsigned char)0xfe);
  do {
    v64u p0 = {Load8u_VSX(src_argb + ((int)ux >> 16) * 4),
               Load8u_VSX(src_argb + ((int)(ux + udx) >> 16) * 4)};
    v64u p1 = {Load8u_VSX(src_argb + ((int)(ux + udx * 2) >> 16) * 4),
               Load8u_VSX(src_argb + ((int)(ux + udx * 3) >> 16) * 4)};
    v8u a = SHUFFLE_VSX((v8u)p0, (v8u)p1, EVEN_ARGB_VSX);
    v8u b = SHUFFLE_VSX((v8u)p0, (v8u)p1, ODD_ARGB_VSX);
    // 2 * f in every byte of each pixel, and 2 * (127 - f).
    v8u f2 = (v8u)vec_mladd(vec_sr(xs, k9), k514, zero);
    v8u nf2 = vec_xor(f2, kfe);
    v16u e = vec_add(vec_mule(a, nf2), vec_mule(b, f2));
    v16u o = vec_add(vec_mulo(a, nf2), vec_mulo(b, f2));
    // The high byte of twice the sum is the sum >> 7, which is the odd byte
    // in little-endian byte order.
    Store16_VSX(SHUFFLE_VSX((v8u)e, (v8u)o, 1, 17, 3, 19, 5, 21, 7, 23, 9, 25,
                            11, 27, 13, 29, 15, 31),
                dst_argb);
    xs = vec_add(xs, dx4);
    ux += udx * 4;
    dst_argb += 16;
    dst_width -= 4;
  } while (dst_width > 0);
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_VSX) && defined(__VSX__) ...
