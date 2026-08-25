#pragma once

#include <array>

#include "accusphgeom/numeric/eft.hpp"

namespace accusphgeom::algorithms {

template <typename T>
using Vec3 = numeric::Vec3<T>;

// Eriksson-formula denominator.
// Returns
//
//   1 + a.b + b.c + c.a.
//
// a.b + b.c + c.a is evaluated as a single length-9 accurate dot product
//
//   [a0,a1,a2, b0,b1,b2, c0,c1,c2] . [b0,b1,b2, c0,c1,c2, a0,a1,a2]
//
// via numeric::accurate_dot_product_fma, so the running compensation carries
// across all three pairwise dot products and is only collapsed to a plain T
// once, at the very end -- rather than computing a.b, b.c, c.a as three
// separately-rounded values and summing those with plain addition, which
// would throw away the compensation right where it matters.
//
// This is the denominator of Eriksson's spherical excess formula
//
//   E = 2 * atan2(numerator, eriksson_denominator(a, b, c)).
template <typename T>
inline T eriksson_denominator(const Vec3<T>& a, const Vec3<T>& b,
                              const Vec3<T>& c) {
  const std::array<T, 9> lhs = {a[0], a[1], a[2], b[0], b[1], b[2],
                                c[0], c[1], c[2]};
  const std::array<T, 9> rhs = {b[0], b[1], b[2], c[0], c[1], c[2],
                                a[0], a[1], a[2]};
  return T(1) + numeric::accurate_dot_product_fma(lhs, rhs);
}

}  // namespace accusphgeom::algorithms
