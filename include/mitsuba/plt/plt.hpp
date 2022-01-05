
/*
    Copyright, PLT authors
*/

#pragma once

#include <array>

#include <mitsuba/core/frame.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/matrix.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/shape.h>

#include "mitsuba/core/constants.h"
#include "mueller.hpp"

MTS_NAMESPACE_BEGIN

struct PLTContext {
    Float Omega, A, sigma_zz;
};

struct RadiancePacket {
    std::array<Vector4,SPECTRUM_SAMPLES> ls{};
    // Matrix2x2 T_x{}, T_y{};     // Shape matrices
    Matrix2x2 T{};
    Frame f{};                  // Reference frame
    Float r{};
    
    inline Spectrum spectrum() const {
        Spectrum s;
        for (auto i=0;i<SPECTRUM_SAMPLES;++i)
            s[i]=ls[i][0];
        return s;
    }

    inline RadiancePacket() noexcept {
        // T_x.setZero();
        // T_y.setZero();
        T.setZero();
    }
    inline RadiancePacket(RadiancePacket&& o) noexcept = default;
    inline RadiancePacket(const RadiancePacket& o) noexcept = default;
    inline RadiancePacket& operator=(RadiancePacket&& o) noexcept = default;
    inline RadiancePacket& operator=(const RadiancePacket& o) noexcept = default;

    inline void rotateShapeMatrices(Float costheta2, Float sintheta2) noexcept {
        // auto x = costheta2*T_x + sintheta2*T_y;
        // T_y = sintheta2*T_x + costheta2*T_y;
        // T_x = x;
    }
    inline void deformShapeMatrices(const Matrix2x2 &Ux, const Matrix2x2 &Uy) noexcept {
        // Matrix2x2 Uxt, Uyt;
        // Ux.transpose(Uxt);
        // Uy.transpose(Uyt);

        // T_x = Ux*T_x*Uxt;
        // T_y = Uy*T_y*Uyt;
    }

    inline void rotateFrame(const Frame &toFrame) noexcept {
        const auto T = MuellerT(f, toFrame);
        for (auto& l : ls)
            l = T*l;
        rotateShapeMatrices(sqr(T.m[1][1]), sqr(T.m[1][2]));

        f = toFrame;
    }
    inline void rotateFrame(const Intersection &its, const Frame &toFrame) noexcept {
        rotateFrame(Frame{ its.toWorld(toFrame.s),
                           its.toWorld(toFrame.t),
                           its.toWorld(toFrame.n) });
    }
    inline void setFrame(const Vector &dir) noexcept {
        this->f = Frame{ dir };
    }
    
    auto& L(std::size_t s) noexcept                 { return ls[s]; }
    const auto& S(std::size_t s) const noexcept     { return ls[s]; }
    const auto Lx(std::size_t s) const noexcept     { return std::max<Float>(0,S(s)[0]+S(s)[1]) / 2; }
    const auto Ly(std::size_t s) const noexcept     { return std::max<Float>(0,S(s)[0]-S(s)[1]) / 2; }
    const auto Ldlp(std::size_t s) const noexcept {
        const auto l = std::sqrt(Lx(s)*Ly(s));
        return l>RCPOVERFLOW ? S(s)[2] / l : .0f;
    }
    const auto Lcp(std::size_t s) const noexcept {
        const auto l = std::sqrt(Lx(s)*Ly(s));
        return l>RCPOVERFLOW ? S(s)[3] / l : .0f;
    }
    const auto Sx(std::size_t s) const noexcept     { return Lx(s) * Vector4{ 1,1,0,0 }; }
    const auto Sy(std::size_t s) const noexcept     { return Ly(s) * Vector4{ 1,-1,0,0 }; }
    const auto Sc(std::size_t s) const noexcept     { return Vector4{ 0,0,S(s)[2],S(s)[3] }; }
    
    void setL(std::size_t s, Float Lx, Float Ly) noexcept {
        const auto dlp = Ldlp(s);
        const auto cp  = Lcp(s);
        const auto sqrtLxLy = std::sqrt(Lx*Ly);
        ls[s] = Vector4{ Lx+Ly,Lx-Ly,dlp*sqrtLxLy,cp*sqrtLxLy };
    }
    
    // const auto Thetax(Float k, Float sigma_zz) const noexcept {
    //     const auto T = sqr(r/k) * T_x;
    //     return Matrix3x3(T.m[0][0], T.m[0][1], 0,
    //                      T.m[0][1], T.m[1][1], 0,
    //                      0,         0,         sigma_zz);
    // } 
    // const auto Thetay(Float k, Float sigma_zz) const noexcept {
    //     const auto T = sqr(r/k) * T_y;
    //     return Matrix3x3(T.m[0][0], T.m[0][1], 0,
    //                      T.m[0][1], T.m[1][1], 0,
    //                      0,         0,         sigma_zz);
    // }
    // const auto Thetac(Float k, Float sigma_zz) const noexcept {
    //     return Float(.5) * (Thetax(k,sigma_zz) + Thetay(k,sigma_zz));
    // }
    
    // const auto invThetax(Float k, Float sigma_zz) const noexcept {
    //     const auto T = sqr(r/k) * T_x;
    //     return Float(1) / (T.m[0][0]*T.m[1][1] - sqr(T.m[0][1])) * 
    //         Matrix3x3( T.m[1][1], -T.m[0][1], 0,
    //                   -T.m[0][1],  T.m[0][0], 0,
    //                    0,          0,         Float(1) / sigma_zz);
    // } 
    // const auto invThetay(Float k, Float sigma_zz) const noexcept {
    //     const auto T = sqr(r/k) * T_y;
    //     return Float(1) / (T.m[0][0]*T.m[1][1] - sqr(T.m[0][1])) * 
    //         Matrix3x3( T.m[1][1], -T.m[0][1], 0,
    //                   -T.m[0][1],  T.m[0][0], 0,
    //                    0,          0,         Float(1) / sigma_zz);
    // } 
    // const auto invThetac(Float k, Float sigma_zz) const noexcept {
    //     const auto T = sqr(r/k) * Float(.5) * (T_x+T_y);
    //     return Float(1) / (T.m[0][0]*T.m[1][1] - sqr(T.m[0][1])) * 
    //         Matrix3x3( T.m[1][1], -T.m[0][1], 0,
    //                   -T.m[0][1],  T.m[0][0], 0,
    //                    0,          0,         Float(1) / sigma_zz);
    // } 
    const auto invTheta(Float k, Float sigma_zz) const noexcept {
        const auto T = sqr(r/k) * this->T;
        return Float(1) / (T.m[0][0]*T.m[1][1] - sqr(T.m[0][1])) * 
            Matrix3x3( T.m[1][1], -T.m[0][1], 0,
                      -T.m[0][1],  T.m[0][0], 0,
                       0,          0,         Float(1) / sigma_zz);
    } 
    
    auto& operator*=(Float f) noexcept {
        for (auto& l : ls)
            l *= f;
        return *this;
    } 
    auto& operator/=(Float f) noexcept {
        for (auto& l : ls)
            l /= f;
        return *this;
    }
    auto& operator*=(const Spectrum &s) noexcept {
        for (auto i=0;i<SPECTRUM_SAMPLES;++i)
            ls[i] *= s[i];
        return *this;
    }
    auto& operator/=(const Spectrum &s) noexcept {
        for (auto i=0;i<SPECTRUM_SAMPLES;++i)
            ls[i] /= s[i];
        return *this;
    }
    
    // In microns^2
    // const auto coherenceArea_x(Float k) const noexcept { return M_PI * sqr(r/k) * std::sqrt(T_x.det()); }
    // const auto coherenceArea_y(Float k) const noexcept { return M_PI * sqr(r/k) * std::sqrt(T_y.det()); }
    const auto coherenceArea(Float k) const noexcept { return M_PI * sqr(r/k) * std::sqrt(T.det()); }
    // Returns the spatial coherence variance in direction v
    const auto coherenceLength(Float k, const Vector3 &v, Float sigma_zz) const noexcept {
        const auto &invT = invTheta(k, sigma_zz);
        return dot(v,(Matrix3x3)invT*v);
    }
    const auto mutualCoherence(Float k, const Vector3 &v, Float sigma_zz) const noexcept {
        const auto &invT = invTheta(k, sigma_zz);
        const auto x = dot(v,(Matrix3x3)invT*v);
        return std::exp(-.5f*x);
    }
    const auto transverseMutualCoherence(Float k, const Vector2 &v) const noexcept {
        Matrix2x2 invT;
        (sqr(r/k) * this->T).invert(invT);
        const auto x = dot(v,invT*v);
        return std::exp(-.5f*x);
    }
    
    const auto begin() const noexcept { return ls.begin(); }
    const auto end() const noexcept { return ls.end(); }
    auto begin() noexcept { return ls.begin(); }
    auto end() noexcept { return ls.end(); }
    const auto size() const noexcept { return ls.size(); }
    const auto& operator[](std::size_t idx) const noexcept { return ls[idx]; }
    auto& operator[](std::size_t idx) noexcept { return ls[idx]; }
    
    bool isValid() const noexcept { return T.m[0][0]>0 && T.m[1][1]>0; }
    // bool isValid() const noexcept { return T_x.m[0][0]>0 && T_x.m[1][1]>0 &&
    //                                        T_y.m[0][0]>0 && T_y.m[1][1]>0; }
};

inline auto sourceLight(Spectrum emission, const PLTContext &ctx) {
    RadiancePacket rad{};
    for (auto i=0;i<SPECTRUM_SAMPLES;++i) 
        rad.ls[i] = Vector4{ emission[i],0,0,0 };
    const auto c = 2 * M_PI * ctx.Omega/ctx.A;
    // rad.T_x = rad.T_y = Matrix2x2(c,0,0,c);
    rad.T = Matrix2x2(c,0,0,c);
    rad.r = 0;
    
    return rad;
}

MTS_NAMESPACE_END
