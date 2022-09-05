/*
    Copyright, PLT authors
*/

#pragma once

#include "mitsuba/core/constants.h"
#include "mitsuba/core/matrix.h"
#include <glm/matrix.hpp>
#include <mitsuba/core/platform.h>
#include <mitsuba/plt/plt.hpp>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/spectrum.h>

#include <glm/glm.hpp>

MTS_NAMESPACE_BEGIN

struct gaussian_fractal_surface {
    float T;
    float gamma;
    float sigma_h2;

    inline auto N(float e) const {
        static constexpr auto c0 = 3.558f;
        static constexpr auto c1 = .5621f;
        const auto c2 = glm::pow<float>((gamma-1.f)/6.447f, 0.0735f);
        static constexpr auto c3 = 3.3016f;
        return 1-c3+c3*(1-c2)*glm::exp(-.5f/c1*(glm::sqrt(e+c0)-glm::sqrt(c0)))+c3*c2;
    }

    inline auto pGF(const glm::vec2 &z, const glm::mat2 &Sigma) const {
        const auto H = 7.f/5.f / (T * (1+gamma)) * glm::mat2(1);
        const auto invHS = glm::inverse(H+Sigma);
        const auto C = (gamma+3)/3.f * (H + 2.f*Sigma);
        const auto invC = glm::inverse(C);
        const auto e = glm::min(1.f, glm::determinant(Sigma) * sqr(T));

        const auto beta = glm::exp(-.5f * glm::dot(z, invC*z));
        const auto g = 
            glm::sqrt(glm::determinant(H)*glm::determinant(invHS)) * 
            glm::exp(-.5f * glm::dot(z, invHS*z));
        const auto f = 1.f/glm::pow(1+T*glm::dot(z,z), (gamma+1.f)/2.f);
        
        return M_PI * (gamma-1) * T / N(e) * (beta*g + (1.f-beta)*f);
    }

    inline auto alpha(const Float costhetai, const Float costhetao) const {
        const auto q = sigma_h2;
        const auto a = -sqr((abs(costhetai)+abs(costhetao)) * q) * Spectrum::ks() * Spectrum::ks();
        return a.exp();
    }

    auto envelopeScattered(const PLTContext &pltCtx, const Vector3 &h) const {
        const auto sigma_min = pltCtx.sigma2_min_um;
        const auto Sigma = 1.f / sigma_min * glm::mat2(1.f);
        
        Spectrum ret;
        for (std::size_t i=0;i<SPECTRUM_SAMPLES;++i) {
            const auto kh = Spectrum::ks()[i]*h;
            ret[i] = pGF({kh.x,kh.y}, Sigma);
        }
        
        return ret;
    }

    auto sampleScattered(const PLTContext &pltCtx, const Vector3 &wi, Sampler &sampler) const {
        const auto sigma_min = pltCtx.sigma2_min_um;
        const auto Sigma = 1.f / sigma_min * glm::mat2(1.f);
        const auto k = Spectrum::ks().average();
        const auto s = sqrt(glm::max(.0f,1-sqr(wi.z)));
        const auto phi_i = s>0 ? glm::atan(wi.y,wi.x) : .0f;

        const auto u2d1 = sampler.next2D();
        const auto u2d2 = sampler.next2D();
        
        const float M = 1.f - glm::pow(1+sqr(k)*T*sqr(1+s), -(gamma-1)/2.f);
        const float f = glm::sqrt(T * (glm::pow(1-M*u2d1.x, -2.f/(gamma-1)) - 1.f)); 

        const auto phi_max = //f==.0f || s==.0f ? 
            float(M_PI);
            //: glm::acos(glm::clamp((sqr(f/k)+sqr(s)-1)/(2.f * (f/k) * s), -1.f,1.f));
        const auto phi_f = phi_i + (2.f*u2d1.y-1)*phi_max;
        const auto vf = f * glm::vec2{ glm::cos(phi_f),glm::sin(phi_f) };

        const auto g = glm::sqrt(-2.f*glm::log(u2d2.x));
        const auto phi_g = 2.f * float(M_PI) * u2d2.y;
        const auto vg = g * glm::sqrt(.5f*(Sigma[0][0]+Sigma[1][1])) * glm::vec2{ glm::cos(phi_g),glm::sin(phi_g) };

        const auto zeta = vf+vg;
        const auto wo = zeta/k - glm::vec2{ wi.x,wi.y };

        const auto z = glm::sqrt(glm::max(.0f,1-glm::dot(wo,wo)));
        return Vector{ wo.x,wo.y,z };
    }

    auto scatteredPdf(const PLTContext &pltCtx, const Vector3 &wi, const Vector3 &wo) const {
        const auto sigma_min = pltCtx.sigma2_min_um;
        const auto Sigma = 1.f / sigma_min * glm::mat2(1.f);
        const auto k = Spectrum::ks().average();

        const auto kh = k*(wi+wo);
        return pGF({kh.x,kh.y}, Sigma);
    }

    inline Float diffract(const Matrix3x3& invSigma, const Matrix3x3 &Q, const Matrix3x3 &Qt, const Vector3 &h) const {
        const auto& invTheta = Q*invSigma*Qt;
        const auto& S = glm::mat2(invTheta.m[0][0], invTheta.m[0][1],
                                       invTheta.m[0][1], invTheta.m[1][1]);
        
        return pGF({h.x,h.y}, S);
    }
};

MTS_NAMESPACE_END
