
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
#include "mueller.h"

MTS_NAMESPACE_BEGIN

struct PLTContext {
    Float Omega, A, sigma_zz;
};

struct RadiancePacket {
    std::array<Vector4,SPECTRUM_SAMPLES> ls{};
    Matrix2x2 T_x{}, T_y{};     // Shape matrices
    Frame f{};                  // Reference frame
    Float r{};
    
    inline Spectrum spectrum() const {
        Spectrum s;
        for (auto i=0;i<SPECTRUM_SAMPLES;++i)
            s[i]=ls[i][0];
        return s;
    }

    inline RadiancePacket() noexcept {
        T_x.setZero();
        T_y.setZero();
    }
    inline RadiancePacket(RadiancePacket&& o) noexcept = default;
    inline RadiancePacket(const RadiancePacket& o) noexcept = default;
    inline RadiancePacket& operator=(RadiancePacket&& o) noexcept = default;
    inline RadiancePacket& operator=(const RadiancePacket& o) noexcept = default;

    inline void rotateShapeMatrices(Float costheta2, Float sintheta2) noexcept {
        auto x = costheta2*T_x + sintheta2*T_y;
        T_y = sintheta2*T_x + costheta2*T_y;
        T_x = x;
    }
    inline void deformShapeMatrices(const Matrix2x2 &Ux, const Matrix2x2 &Uy) noexcept {
        Matrix2x2 Uxt, Uyt;
        Ux.transpose(Uxt);
        Uy.transpose(Uyt);

        T_x = Ux*T_x*Uxt;
        T_y = Uy*T_y*Uyt;
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
    
    auto& L(std::size_t s) noexcept                 { return ls[s]; }
    const auto& S(std::size_t s) const noexcept     { return ls[s]; }
    const auto Lx(std::size_t s) const noexcept     { return (S(s)[0]+S(s)[1]) / 2; }
    const auto Ly(std::size_t s) const noexcept     { return (S(s)[0]-S(s)[1]) / 2; }
    const auto Ldlp(std::size_t s) const noexcept   { return S(s)[2] / std::sqrt(Lx(s)*Ly(s)); }
    const auto Lcp(std::size_t s) const noexcept    { return S(s)[3] / std::sqrt(Lx(s)*Ly(s)); }
    const auto Sx(std::size_t s) const noexcept     { return Lx(s) * Vector4{ 1,1,0,0 }; }
    const auto Sy(std::size_t s) const noexcept     { return Ly(s) * Vector4{ 1,-1,0,0 }; }
    const auto Sc(std::size_t s) const noexcept     { return Vector4{ 0,0,S(s)[2],S(s)[3] }; }
    
    const auto size() const noexcept { return ls.size(); }
    const auto& operator[](std::size_t idx) const noexcept { return ls[idx]; }
    auto& operator[](std::size_t idx) noexcept { return ls[idx]; }
    
    bool isValid() const noexcept { return T_x.m[0][0]>0 && T_x.m[1][1]>0 &&
                                           T_y.m[0][0]>0 && T_y.m[1][1]>0; }
};

inline auto sourceLight(Vector dir, Spectrum emission, const PLTContext &ctx) {
    RadiancePacket rad{};
    rad.f = Frame{ dir };
    rad.r = 0;
    const auto c = ctx.Omega/(2*M_PI*ctx.A);
    for (auto i=0;i<SPECTRUM_SAMPLES;++i) 
        rad.ls[i] = Vector4{ emission[i],0,0,0 };
    rad.T_x = rad.T_y = Matrix2x2(c,0,0,c);
    
    return rad;
}

MTS_NAMESPACE_END
