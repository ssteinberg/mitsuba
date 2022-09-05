
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

struct fresnel_spm_order1_data {
    float mr[2][2];
    float mt[2][2];
    std::complex<float> kai1[2];
    std::complex<float> kai2[2];
    std::complex<float> kai3[2];
    std::complex<float> beta1[2];
    std::complex<float> beta2[2];
    std::complex<float> beta3[2];

    float R,T;
};
inline fresnel_spm_order1_data fresnel_spm_order1(float cos_theta_i, float cos_theta_o, const float phi_i, const float phi_o, std::complex<float> eta) {
    using namespace std;
    using c_t = complex<float>;

    if (cos_theta_i<.0f) 
        eta = c_t(1) / eta;
    const float reta = real(eta);
    cos_theta_i = abs(cos_theta_i);
    cos_theta_o = abs(cos_theta_o);
    const float sin_theta_i = sqrt(max(.0f, 1.f - sqr(cos_theta_i)));
    const float sin_theta_o = sqrt(max(.0f, 1.f - sqr(cos_theta_o)));
    
    const float phi = phi_o - phi_i + M_PI;
    const float cos_phi = cos(phi);
    const float sin_phi = sin(phi);
    const c_t eta2 = eta*eta;

    const c_t si = sqrt(eta2 - c_t(sqr(sin_theta_i)));
    const c_t ss = sqrt(eta2 - c_t(sqr(sin_theta_o)));
    const c_t st = sqrt(c_t(1) - eta2*sqr(sin_theta_o));
    const c_t qi = c_t(cos_theta_i) - si;
    const c_t qs = c_t(cos_theta_o) - ss;
    const c_t qt = eta*cos_theta_o - st;
    const c_t ti = c_t(sqr(sin_theta_i)) + si * cos_theta_i;
    const c_t ts = c_t(sqr(sin_theta_o)) + ss * cos_theta_o;
    const c_t tt = eta*sqr(sin_theta_o) + st * cos_theta_o;
    const c_t wi = -sqr(si) + si * cos_theta_i;
    const c_t ws = -sqr(ss) + ss * cos_theta_o;
    const c_t wt = -sqr(st) + eta*st * cos_theta_o;
    const c_t s  = eta2 * sin_theta_i*sin_theta_o - si*ss*cos_phi;
    const c_t t  = eta  * sin_theta_i*sin_theta_o + si*st*cos_phi;

    fresnel_spm_order1_data d;

    d.R = cos_theta_i / norm(c_t(cos_theta_i) + si);
    d.T = cos_theta_i * reta*sqrt(reta) / sqr(sqr(reta)-1);

    for (int j=0;j<2;++j)
    for (int l=0;l<2;++l) {
        const float ll = l==0?1.f:-1.f;
        const float jj = j==0?1.f:-1.f;
        d.mr[j][l] = ll * norm(qs*s/(ti*ts)) +
                     norm(qs) * sqr(cos_phi) +
                     jj * (norm(ws/ts) + ll*norm(qs*si/ti)) * sqr(sin_phi);
        d.mt[j][l] = ll * norm(qi*qt*t/(ti*tt)) +
                     norm(qi*qt) * sqr(cos_phi) +
                     jj * (norm(qi*wt/tt) + ll*norm(qt*wi/ti)) * sqr(sin_phi);
    }
    for (int j=0;j<2;++j) {
        const float jj = j==0?1.f:-1.f;

        d.kai1[j] = jj*(qs*s*conj(ws)*sin_phi/(ti*norm(ts))) + si*norm(qs)*cos_phi*sin_phi/ti;
        d.kai2[j] = (jj*si*conj(s)*norm(qs/ti) + qs*conj(ws)*cos_phi) / conj(ts) * sin_phi;
        d.kai3[j] = jj*qs*si*conj(ws)*sqr(sin_phi)/(ti*conj(ts)) + s*norm(qs)*cos_phi/(ti*ts);

        d.beta1[j] = (jj*qt*t*norm(qi/tt)*conj(wt) + wi*norm(qt)*conj(qi)*cos_phi) / ti * sin_phi;
        d.beta2[j] = (jj*qi*t*norm(qt/ti)*conj(wi) + wt*norm(qi)*conj(qt)*cos_phi) / tt * sin_phi;
        d.beta3[j] = jj*qi*wt*conj(qt)*conj(wi)*sqr(sin_phi)/(tt*conj(ti)) + t*norm(qi*qt)*cos_phi/(tt*ti);
    }

    return d;
}

inline Matrix4x4 MuellerFresnelRspm1(float cos_theta_i, float cos_theta_o, const float phi_i, const float phi_o, std::complex<float> eta) noexcept {
    using namespace std;

    const auto d = fresnel_spm_order1(cos_theta_i, cos_theta_o, phi_i, phi_o, eta);

    return d.R * Matrix4x4(
        .5f*d.mr[0][0],  .5f*d.mr[1][1],  real(d.kai2[0]), imag(d.kai2[0]),
        .5f*d.mr[0][1],  .5f*d.mr[1][0],  real(d.kai2[1]), imag(d.kai2[1]),
        real(d.kai1[0]), real(d.kai1[1]), real(d.kai3[0]), -imag(d.kai3[1]),
        imag(d.kai1[0]), imag(d.kai1[1]), imag(d.kai3[0]), real(d.kai3[1])
    );
}

inline Matrix4x4 MuellerFresnelTspm1(float cos_theta_i, float cos_theta_o, const float phi_i, const float phi_o, float eta) noexcept {
    using namespace std;

    const auto d = fresnel_spm_order1(cos_theta_i, cos_theta_o, phi_i, phi_o, std::complex<float>(eta,.0f));

    return d.T * Matrix4x4(
        .5f*d.mt[0][0],  .5f*d.mt[1][1],  -real(d.beta2[1]), imag(d.beta2[1]),
        .5f*d.mt[0][1],  .5f*d.mt[1][0],  -real(d.beta2[0]), imag(d.beta2[0]),
        real(d.beta1[1]), real(d.beta1[0]), real(d.beta3[1]), -imag(d.beta3[1]),
        imag(d.beta1[1]), imag(d.beta1[0]), imag(d.beta3[0]), real(d.beta3[0])
    );
}

inline Matrix4x4 MuellerFresnelspm1(float cos_theta_i, float cos_theta_o, const float phi_i, const float phi_o, std::complex<float> eta, bool isReflection) noexcept {
    return isReflection ?
        MuellerFresnelRspm1(cos_theta_i, cos_theta_o, phi_i, phi_o, eta) :
        MuellerFresnelTspm1(cos_theta_i, cos_theta_o, phi_i, phi_o, std::real(eta));
}


MTS_NAMESPACE_END
