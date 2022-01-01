/*
    Copyright, PLT authors
*/

#pragma once

#include <mitsuba/core/platform.h>
#include <mitsuba/plt/plt.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/spectrum.h>

#include <boost/math/distributions/normal.hpp>

MTS_NAMESPACE_BEGIN

namespace gaussianSurface {

inline auto alpha(const Float costhetai, const Float q) {
    const auto a = -sqr(costhetai * q) * Spectrum::ks() * Spectrum::ks();
    return a.exp();
}

auto envelopeScattered(Float sigma2, const PLTContext &pltCtx, const Vector3 &h) {
    const auto sigma_min = pltCtx.sigma_zz * 1e+6f; // metres to micron
    const auto w = Float(1) / sigma_min + Float(1) / sigma2;
    
    const auto dist = boost::math::normal{ Float(0), w };
    Spectrum gx, gy;
    for (std::size_t i=0;i<SPECTRUM_SAMPLES;++i) {
        const auto kh = Spectrum::ks()[i]*h;
        gx[i] = boost::math::pdf(dist, kh.x);
        gy[i] = boost::math::pdf(dist, kh.y);
    }
    
    return gx*gy;
}

auto sampleScattered(Float sigma2, const PLTContext &pltCtx, const Vector3 &wi, Sampler &sampler) {
    const auto sigma_min = pltCtx.sigma_zz * 1e+6f; // metres to micron
    const auto w = Float(1) / sigma_min + Float(1) / sigma2;
    const auto k = Spectrum::ks().average();
    
    return warp::squareToTruncatedGaussian(std::sqrt(w)/k, -Point2{ wi.x,wi.y }, sampler);
}

auto scatteredPdf(Float sigma2, const PLTContext &pltCtx, const Vector3 &wi, const Vector3 &wo) {
    const auto sigma_min = pltCtx.sigma_zz * 1e+6f; // metres to micron
    const auto w = Float(1) / sigma_min + Float(1) / sigma2;
    const auto k = Spectrum::ks().average();
    
    return warp::squareToTruncatedGaussianPdf(std::sqrt(w)/k, -Point2{ wi.x,wi.y }, wo);
}

inline auto diffract(const Matrix3x3& invSigma, Float sigma2, const Matrix3x3 &Q, const Matrix3x3 &Qt, const Vector3 &h) {
    const auto& invTheta = Q*invSigma*Qt;
    
    const Matrix2x2& S = (1.f/sigma2) * Matrix2x2(1,0,0,1) + 
                         Matrix2x2(invTheta.m[0][0], invTheta.m[0][1],
                                   invTheta.m[0][1], invTheta.m[1][1]);
    Matrix2x2 invS;
    if (!S.invert2x2(invS))
        return Float(0);

    const auto& h2 = Vector2{ h.x,h.y };
    return sqr(2 * M_PI) / std::sqrt(S.det()) * 
            std::exp(-dot(h2,invS*h2)/2);
}

}

MTS_NAMESPACE_END
