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
typedef __vector signed char v8s;
typedef __vector unsigned short v16u;
typedef __vector signed short v16s;
typedef __vector unsigned int v32u;
typedef __vector unsigned long long v64u;

static __inline v8u Load16_VSX(const uint8_t* p) {
  return vec_xl(0, p);
}

// Loads 8 bytes into the first half of the vector.
static __inline v8u Load8_VSX(const uint8_t* p) {
  unsigned long long v;
  memcpy(&v, p, 8);
  return (v8u)vec_splats(v);
}

static __inline void Store16_VSX(v8u v, uint8_t* p) {
  vec_xst(v, 0, p);
}

// Constant byte shuffle of two v8u. The indices count elements in memory
// order on both byte orders.
#if defined(__clang__) || __GNUC__ >= 12
#define SHUFFLE_VSX(a, b, ...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
#define SHUFFLE_VSX(a, b, ...) __builtin_shuffle(a, b, (v8u){__VA_ARGS__})
#endif

// Each byte of the first or last 8, twice.
#define DUP2_LO_VSX 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7
#define DUP2_HI_VSX 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15
// Each of the first or second 4 bytes, four times.
#define DUP4_LO_VSX 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
#define DUP4_HI_VSX 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7

// a * b >> 16 for each lane. The shuffle picks the high half of each 32-bit
// product, which is the odd 16-bit half in little-endian byte order.
static __inline v16u MulHi16_VSX(v16u a, v16u b) {
  v8u e = (v8u)vec_mule(a, b);
  v8u o = (v8u)vec_mulo(a, b);
  return (v16u)SHUFFLE_VSX(e, o, 2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14,
                           15, 30, 31);
}

// Stores 8 ARGB pixels from B and G of each pixel in bg, and R and A in ra.
static __inline void StoreARGB8_VSX(v8u bg, v8u ra, uint8_t* dst_argb) {
  Store16_VSX(SHUFFLE_VSX(bg, ra, 0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3,
                          11, 19, 27),
              dst_argb);
  Store16_VSX(SHUFFLE_VSX(bg, ra, 4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30,
                          7, 15, 23, 31),
              dst_argb + 16);
}

// Stores 16 opaque gray ARGB pixels.
static __inline void StoreGray_VSX(v8u y, uint8_t* dst_argb) {
  v8u alpha = vec_splats((unsigned char)0xff);
  Store16_VSX(
      SHUFFLE_VSX(y, alpha, 0, 0, 0, 16, 1, 1, 1, 16, 2, 2, 2, 16, 3, 3, 3, 16),
      dst_argb);
  Store16_VSX(
      SHUFFLE_VSX(y, alpha, 4, 4, 4, 16, 5, 5, 5, 16, 6, 6, 6, 16, 7, 7, 7, 16),
      dst_argb + 16);
  Store16_VSX(SHUFFLE_VSX(y, alpha, 8, 8, 8, 16, 9, 9, 9, 16, 10, 10, 10, 16,
                          11, 11, 11, 16),
              dst_argb + 32);
  Store16_VSX(SHUFFLE_VSX(y, alpha, 12, 12, 12, 16, 13, 13, 13, 16, 14, 14, 14,
                          16, 15, 15, 15, 16),
              dst_argb + 48);
}

// YuvConstants splatted to 16-bit lanes.
struct YuvCoefs_VSX {
  v16s ub, ug, vg, vr, yb;
  v16u yg;
};

static __inline void LoadYuvCoefs_VSX(const struct YuvConstants* yuvconstants,
                                      struct YuvCoefs_VSX* c) {
  c->ub = vec_splats((short)yuvconstants->kUVToB[0]);
  c->ug = vec_splats((short)yuvconstants->kUVToG[0]);
  c->vg = vec_splats((short)yuvconstants->kUVToG[1]);
  c->vr = vec_splats((short)yuvconstants->kUVToR[1]);
  c->yg = vec_splats((unsigned short)yuvconstants->kYToRgb[0]);
  c->yb = vec_splats((short)yuvconstants->kYBiasToRgb[0]);
}

// The YUV rows widen Y, U and V by duplicating bytes, so each 16-bit lane
// holds its value times 0x0101.

// u - 128 from a lane holding u * 0x0101.
static __inline v16s UV16_VSX(v8u uv) {
  return vec_sub((v16s)vec_sr((v16u)uv, vec_splats((unsigned short)8)),
                 vec_splats((short)128));
}

// (y * 0x0101 * yg >> 16) + yb from a lane holding y * 0x0101.
static __inline v16s Y16_VSX(v8u y, const struct YuvCoefs_VSX* c) {
  return vec_add((v16s)MulHi16_VSX((v16u)y, c->yg), c->yb);
}

// U and V contributions to B, G and R for 8 pixels, or for 8 pairs of
// pixels in 4:2:2.
struct UVTerms_VSX {
  v16s b, g, r;
};

static __inline void UVTerms_VSX(v8u u,
                                 v8u v,
                                 const struct YuvCoefs_VSX* c,
                                 struct UVTerms_VSX* t) {
  v16s zero = vec_splats((short)0);
  v16s ui = UV16_VSX(u);
  v16s vi = UV16_VSX(v);
  t->b = vec_mladd(ui, c->ub, zero);
  t->g = vec_mladd(ui, c->ug, vec_mladd(vi, c->vg, zero));
  t->r = vec_mladd(vi, c->vr, zero);
}

// Converts 8 pixels and stores them. a is 8 alpha bytes in the first half,
// or null for opaque pixels.
static __inline void YuvToARGB8_VSX(v16s y1,
                                    v16s tb,
                                    v16s tg,
                                    v16s tr,
                                    const v8u* a,
                                    uint8_t* dst_argb) {
  v16u six = vec_splats((unsigned short)6);
  v16s b = vec_sra(vec_adds(y1, tb), six);
  v16s g = vec_sra(vec_subs(y1, tg), six);
  v16s r = vec_sra(vec_adds(y1, tr), six);
  v8u ra = a ? (v8u)vec_mergeh((v64u)vec_packsu(r, r), (v64u)*a)
             : vec_packsu(r, vec_splats((short)255));
  StoreARGB8_VSX(vec_packsu(b, g), ra, dst_argb);
}

// Converts and stores 16 pixels that share 8 U and V in pairs.
static __inline void Yuv422ToARGB16_VSX(v8u y,
                                        v8u u,
                                        v8u v,
                                        const v8u* a,
                                        const struct YuvCoefs_VSX* c,
                                        uint8_t* dst_argb) {
  struct UVTerms_VSX t;
  UVTerms_VSX(SHUFFLE_VSX(u, u, DUP2_LO_VSX), SHUFFLE_VSX(v, v, DUP2_LO_VSX), c,
              &t);
  v8u a1 =
      a ? (v8u)vec_mergel((v64u)*a, (v64u)*a) : vec_splats((unsigned char)0);
  YuvToARGB8_VSX(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_LO_VSX), c),
                 vec_mergeh(t.b, t.b), vec_mergeh(t.g, t.g),
                 vec_mergeh(t.r, t.r), a, dst_argb);
  YuvToARGB8_VSX(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_HI_VSX), c),
                 vec_mergel(t.b, t.b), vec_mergel(t.g, t.g),
                 vec_mergel(t.r, t.r), a ? &a1 : 0, dst_argb + 32);
}

// Converts and stores 16 pixels with their own U and V.
static __inline void Yuv444ToARGB16_VSX(v8u y,
                                        v8u u,
                                        v8u v,
                                        const v8u* a,
                                        const struct YuvCoefs_VSX* c,
                                        uint8_t* dst_argb) {
  struct UVTerms_VSX t0, t1;
  UVTerms_VSX(SHUFFLE_VSX(u, u, DUP2_LO_VSX), SHUFFLE_VSX(v, v, DUP2_LO_VSX), c,
              &t0);
  UVTerms_VSX(SHUFFLE_VSX(u, u, DUP2_HI_VSX), SHUFFLE_VSX(v, v, DUP2_HI_VSX), c,
              &t1);
  v8u a1 =
      a ? (v8u)vec_mergel((v64u)*a, (v64u)*a) : vec_splats((unsigned char)0);
  YuvToARGB8_VSX(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_LO_VSX), c), t0.b, t0.g, t0.r,
                 a, dst_argb);
  YuvToARGB8_VSX(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_HI_VSX), c), t1.b, t1.g, t1.r,
                 a ? &a1 : 0, dst_argb + 32);
}

void I422ToARGBRow_VSX(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    Yuv422ToARGB16_VSX(Load16_VSX(src_y), Load8_VSX(src_u), Load8_VSX(src_v), 0,
                       &c, dst_argb);
    src_y += 16;
    src_u += 8;
    src_v += 8;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I422AlphaToARGBRow_VSX(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            const uint8_t* src_a,
                            uint8_t* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    v8u a = Load16_VSX(src_a);
    Yuv422ToARGB16_VSX(Load16_VSX(src_y), Load8_VSX(src_u), Load8_VSX(src_v),
                       &a, &c, dst_argb);
    src_y += 16;
    src_u += 8;
    src_v += 8;
    src_a += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I444ToARGBRow_VSX(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    Yuv444ToARGB16_VSX(Load16_VSX(src_y), Load16_VSX(src_u), Load16_VSX(src_v),
                       0, &c, dst_argb);
    src_y += 16;
    src_u += 16;
    src_v += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I444AlphaToARGBRow_VSX(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            const uint8_t* src_a,
                            uint8_t* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    v8u a = Load16_VSX(src_a);
    Yuv444ToARGB16_VSX(Load16_VSX(src_y), Load16_VSX(src_u), Load16_VSX(src_v),
                       &a, &c, dst_argb);
    src_y += 16;
    src_u += 16;
    src_v += 16;
    src_a += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I400ToARGBRow_VSX(const uint8_t* src_y,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VSX c;
  v16u six = vec_splats((unsigned short)6);
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    v8u y = Load16_VSX(src_y);
    v16s lo = vec_sra(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_LO_VSX), &c), six);
    v16s hi = vec_sra(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_HI_VSX), &c), six);
    StoreGray_VSX(vec_packsu(lo, hi), dst_argb);
    src_y += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void J400ToARGBRow_VSX(const uint8_t* src_y, uint8_t* dst_argb, int width) {
  do {
    v8u y0 = Load16_VSX(src_y);
    v8u y1 = Load16_VSX(src_y + 16);
    StoreGray_VSX(y0, dst_argb);
    StoreGray_VSX(y1, dst_argb + 64);
    src_y += 32;
    dst_argb += 128;
    width -= 32;
  } while (width > 0);
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_VSX) && defined(__VSX__) ...
