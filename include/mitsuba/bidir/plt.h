
/*
    Copyright, PLT authors
*/

#pragma once

#include <array>

#include <mitsuba/core/vector.h>
#include <mitsuba/core/matrix.h>
#include <mitsuba/core/spectrum.h>

MTS_NAMESPACE_BEGIN

struct PLTContext {
    Float Omega, A, sigma_zz;
};

struct RadiancePacket {
    struct WavePacket {
        Vector4 l;              // Stokes parameters vector
        Matrix2x2 T_x, T_y;     // Shape matrices
    };

    std::array<WavePacket,SPECTRUM_SAMPLES> Ls{};
    Vector3 x,y;    // Reference frame
    
    Spectrum spectrum() const {
        Spectrum s;
        for (auto i=0;i<SPECTRUM_SAMPLES;++i)
            s[i]=Ls[i].l[0];
        return s;
    }

    RadiancePacket& operator=(const RadiancePacket& o) noexcept {
        Ls = o.Ls;
        return *this;
    }
};

MTS_NAMESPACE_END
