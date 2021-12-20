
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

    inline void rotateFrame(const Frame &toFrame) noexcept {
        const auto T = MuellerT(f, toFrame);
        for (auto& L : Ls) {
            L.l = T*L.l;
            rotateShapeMatrices(sqr(T.m[1][1]), sqr(T.m[1][2]), L);
        }

        f = toFrame;
    }

    inline void diffract() noexcept {

    }
};

struct ImportancePacket {
    std::array<Vector4,SPECTRUM_SAMPLES> ls{};
    Frame f{};    // Reference frame
    
    inline Spectrum spectrum() const {
        Spectrum s;
        for (auto i=0;i<SPECTRUM_SAMPLES;++i)
            s[i]=ls[i][0];
        return s;
    }

    inline ImportancePacket& operator=(const ImportancePacket& o) noexcept {
        ls=o.ls;
        f=o.f;
        return *this;
    }
};

MTS_NAMESPACE_END
