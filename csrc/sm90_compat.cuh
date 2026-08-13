#pragma once
// SM90/H20 port: parse stubs for Blackwell MX-scale types. The BF16-only
// build never instantiates the USE_MXFP8 paths; these stubs keep the
// discarded constexpr branches parsing with identical 1-byte layout math.
#if defined(KITTENS_SM90)
namespace kittens {
using fp8e8m0 = uint8;
template<int _height, int _width, bool _swizzle = true, int _swizzle_bytes = 0>
using st_fp8e8m0 = st<uint8, _height, _width, _swizzle, _swizzle_bytes>;
} // namespace kittens
#endif
