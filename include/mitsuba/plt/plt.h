
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

#include "mueller.h"

MTS_NAMESPACE_BEGIN

struct PLTContext {
    Float Omega, A, sigma_zz;
};

struct RadiancePacket {
    struct WavePacket {
        Vector4 l;              // Stokes parameters vector
        Matrix2x2 T_x, T_y;     // Shape matrices
    };
    inline void rotateShapeMatrices(Float costheta2, Float sintheta2, WavePacket &wp) const noexcept {
        auto T_x = costheta2*wp.T_x + sintheta2*wp.T_y;
        wp.T_y = sintheta2*wp.T_x + costheta2*wp.T_y;
        wp.T_x = T_x;
    }
    inline void deformShapeMatrices(const Matrix2x2 &Ux, const Matrix2x2 &Uy, WavePacket &wp) const noexcept {
        Matrix2x2 Uxt, Uyt;
        Ux.transpose(Uxt);
        Uy.transpose(Uyt);

        wp.T_x = Ux*wp.T_x*Uxt;
        wp.T_y = Uy*wp.T_y*Uyt;
    }

    std::array<WavePacket,SPECTRUM_SAMPLES> Ls{};
    Frame f{};    // Reference frame
    Float r{};
    
    inline Spectrum spectrum() const {
        Spectrum s;
        for (auto i=0;i<SPECTRUM_SAMPLES;++i)
            s[i]=Ls[i].l[0];
        return s;
    }

    inline RadiancePacket& operator=(const RadiancePacket& o) noexcept {
        Ls=o.Ls;
        f=o.f;
        r=o.r;
        return *this;
    }

    inline void rotateFrame(const Intersection &its, Frame toFrame) noexcept {
        toFrame.n = its.toWorld(toFrame.n);
        toFrame.s = its.toWorld(toFrame.s);
        toFrame.t = its.toWorld(toFrame.t);

        const auto T = MuellerT(f, toFrame);
        for (auto& L : Ls) {
            L.l = T*L.l;
            rotateShapeMatrices(sqr(T.m[1][1]), sqr(T.m[1][2]), L);
        }

        f = toFrame;
    }
    
    auto& L(std::size_t s) noexcept                 { return Ls[s].l; }
    const auto& S(std::size_t s) const noexcept     { return Ls[s].l; }
    const auto Lx(std::size_t s) const noexcept     { return (S(s)[0]+S(s)[1]) / 2; }
    const auto Ly(std::size_t s) const noexcept     { return (S(s)[0]-S(s)[1]) / 2; }
    const auto Ldlp(std::size_t s) const noexcept   { return S(s)[2] / std::sqrt(Lx(s)*Ly(s)); }
    const auto Lcp(std::size_t s) const noexcept    { return S(s)[3] / std::sqrt(Lx(s)*Ly(s)); }
    const auto Sx(std::size_t s) const noexcept     { return Lx(s) * Vector4{ 1,1,0,0 }; }
    const auto Sy(std::size_t s) const noexcept     { return Ly(s) * Vector4{ 1,-1,0,0 }; }
    const auto Sc(std::size_t s) const noexcept     { return Vector4{ 0,0,S(s)[2],S(s)[3] }; }
    
    const auto size() const noexcept { return Ls.size(); }
    const auto& operator[](std::size_t idx) const noexcept { return Ls[idx]; }
    auto& operator[](std::size_t idx) noexcept { return Ls[idx]; }
};

MTS_NAMESPACE_END
