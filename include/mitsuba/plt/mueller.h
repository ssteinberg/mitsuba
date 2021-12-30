
/*
    Copyright, PLT authors
*/

#pragma once

#include <complex>
#include <mitsuba/core/math.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/matrix.h>
#include <mitsuba/core/util.h>

#include <cmath>
#include <cassert>

MTS_NAMESPACE_BEGIN

// Mueller rotation matrix between reference frames
inline Matrix4x4 MuellerT(const Frame &f1, const Frame &f2) {
    auto p = f2.s*(1 - dot(f1.n,f2.s));
    Float cost{};
    if (dot(p,p)>1e-5) {
        p = normalize(p);
        cost = dot(p,f1.s);
    }
    else {
        p = f2.t*(1 - dot(f1.n,f2.t));
        p = normalize(p);
        cost = dot(p,f1.t);
    }
    
    const auto sint2 = 1-cost*cost;
    const auto sint  = math::safe_sqrt(sint2);
    const auto rx    = 2*sint*cost;
    const auto ry    = 1-2*sint2;
    
    Matrix4x4 T{ Float(0) };
    T.m[0][0] = T.m[3][3] = 1;
    T.m[1][1] = T.m[2][2] = rx;
    T.m[1][2] = -ry;
    T.m[2][1] = ry;
    return T;
}

// Mueller Fresnel matrices
template <typename T>
auto MuellerFresnel(const T &fs, const T &fp) noexcept {
    const auto Rp = sqr(std::abs(fp));
    const auto Rs = sqr(std::abs(fs));
    const auto m00 = (Rp+Rs)/2;
    const auto m01 = (Rp-Rs)/2;
    const auto m22 = std::real(fs*std::conj(fp));
    const auto m23 = std::imag(fs*std::conj(fp));
    return Matrix4x4(m00,m01, 0,  0,
                     m01,m00, 0,  0,
                     0,  0,   m22,m23,
                     0,  0,  -m23,m22);
}
inline Matrix4x4 MuellerFresnelRConductor(Float cosThetaI, const std::complex<Float>& eta) noexcept {
    std::complex<Float> rs,rp;
    fresnel_conductor(cosThetaI, eta, rs, rp);

    return MuellerFresnel(rs, rp);
}
inline Matrix4x4 MuellerFresnelRDielectric(Float cosThetaI, Float eta) noexcept {
    Float cosThetaT, rs,rp, ts,tp;
    fresnel_dielectric(cosThetaI, eta, cosThetaT, rs, rp, ts, tp);

    return MuellerFresnel(rs, rp);
}
inline Matrix4x4 MuellerFresnelTDielectric(Float cosThetaI, Float &cosThetaT, Float eta) noexcept {
    Float rs,rp, ts,tp;
    fresnel_dielectric(cosThetaI, eta, cosThetaT, rs, rp, ts, tp);

    return MuellerFresnel(ts, tp);
}

MTS_NAMESPACE_END
