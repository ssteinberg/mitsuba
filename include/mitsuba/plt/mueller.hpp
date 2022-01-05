
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

// Mueller linear polarizer
template <typename T>
auto MuellerPolarizer(T theta) noexcept {
    T c,s;
    math::sincos(theta,&s,&c);

    Matrix4x4 P;
    P.setZero();
    P.m[0][0] = 1;
    P.m[0][1] = P.m[1][0] = c;
    P.m[0][2] = P.m[2][0] = s;
    P.m[1][2] = P.m[2][1] = c*s;
    P.m[1][1] = c*c;
    P.m[2][2] = s*s;

    return P / T(2);
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
template <typename T>
auto invOneMinusMuellerFresnel(const T &fs, const T &fp) noexcept {
    const auto Rp = sqr(std::abs(fp));
    const auto Rs = sqr(std::abs(fs));
    const auto m00 = (Rp+Rs)/2;
    const auto m01 = (Rp-Rs)/2;
    const auto m22 = std::real(fs*std::conj(fp));
    const auto m23 = std::imag(fs*std::conj(fp));
    
    const auto minor1 = Matrix2x2(1,0,0,1) - Matrix2x2(m00,m01, m01,m00);
    const auto minor2 = Matrix2x2(1,0,0,1) - Matrix2x2(m22,m23,-m23,m22);
    Matrix2x2 im1, im2;
    if (!minor1.invert(im1) || !minor2.invert(im2)) {
        assert(false);
        Matrix4x4 z;
        z.setZero();
        return z;
    }
    
    return Matrix4x4(im1.m[0][0],im1.m[0][1], 0,          0,
                     im1.m[0][1],im1.m[0][0], 0,          0,
                     0,          0,           im2.m[0][0],im2.m[0][1],
                     0,          0,           im2.m[1][0],im2.m[0][0]);
}
inline Matrix4x4 MuellerFresnelRConductor(Float cosThetaI, const std::complex<Float>& eta) noexcept {
    std::complex<Float> rs,rp;
    fresnel_conductor(cosThetaI, eta, rs, rp);

    return MuellerFresnel(rs, rp);
}
inline Matrix4x4 MuellerFresnelRDielectric(Float cosThetaI, Float eta) noexcept {
    Float rs,rp, ts,tp;
    fresnel_dielectric(cosThetaI, eta, rs, rp, ts, tp);

    return MuellerFresnel(rs, rp);
}
inline Matrix4x4 invOneMinusMuellerFresnelRDielectric(Float cosThetaI, Float eta) noexcept {
    Float rs,rp, ts,tp;
    fresnel_dielectric(cosThetaI, eta, rs, rp, ts, tp);

    return invOneMinusMuellerFresnel(rs, rp);
}
inline Matrix4x4 MuellerFresnelTDielectric(Float cosThetaI, Float eta) noexcept {
    Float rs,rp, ts,tp;
    fresnel_dielectric(cosThetaI, eta, rs, rp, ts, tp);

    return MuellerFresnel(ts, tp);
}
inline Matrix4x4 MuellerFresnelDielectric(Float cosThetaI, Float eta, bool reflection) noexcept {
    return reflection ?
        MuellerFresnelRDielectric(cosThetaI, eta) :
        MuellerFresnelTDielectric(cosThetaI, eta);
}

MTS_NAMESPACE_END
