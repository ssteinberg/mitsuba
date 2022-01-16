/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <cmath>

#include <mitsuba/core/fresolver.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/plt/plt.hpp>
#include <mitsuba/plt/mueller.hpp>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/core/constants.h>

#include <mitsuba/plt/gaussianSurface.hpp>

#include "ior.h"
#include "mitsuba/render/common.h"

MTS_NAMESPACE_BEGIN

class Grating : public BSDF {
public:
    Grating(const Properties &props) : BSDF(props) {
        ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();

        m_specularReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("specularReflectance", Spectrum(1.0f)));

        std::string materialName = props.getString("material", "Cu");

        Spectrum intEta, intK;
        if (boost::to_lower_copy(materialName) == "none") {
            intEta = Spectrum(0.0f);
            intK = Spectrum(1.0f);
        } else {
            intEta.fromContinuousSpectrum(InterpolatedSpectrum(
                fResolver->resolve("data/ior/" + materialName + ".eta.spd")));
            intK.fromContinuousSpectrum(InterpolatedSpectrum(
                fResolver->resolve("data/ior/" + materialName + ".k.spd")));
        }

        Float extEta = lookupIOR(props, "extEta", "air");

        m_eta = props.getSpectrum("eta", intEta) / extEta;
        m_k   = props.getSpectrum("k", intK) / extEta;

        // Grating paramters
        std::vector<std::string> lobes = tokenize(props.getString("lobes", ""), " ,;");
        if (lobes.size() == 0)      Log(EError, "No lobes were supplied!");
        m_lobes.resize(lobes.size());
        char *end_ptr = NULL;
        for (size_t i=0; i<lobes.size(); ++i) {
            Float lobe = (Float) strtod(lobes[i].c_str(), &end_ptr);
            if (*end_ptr != '\0')   SLog(EError, "Could not parse the diffraction lobes!");
            if (lobe < 0)           SLog(EError, "Invalid diffraction lobe!");
            m_lobes[i] = std::max<Float>(0,lobe);
        }

        m_pitch = props.getFloat("pitch");
        m_q = props.getFloat("q",0.5);
        m_gratingDirU = props.getBoolean("gratingInDirectionU", true);
    }

    Grating(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_eta = Spectrum(stream);
        m_k = Spectrum(stream);
        
        m_lobes.resize(stream->readSize());
        for (size_t i=0; i<m_lobes.size(); ++i)
            m_lobes[i] = stream->readFloat();
        m_pitch = stream->readFloat();
        m_q = stream->readFloat();
        m_gratingDirU = stream->readBool();

        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        manager->serialize(stream, m_specularReflectance.get());
        m_eta.serialize(stream);
        m_k.serialize(stream);

        stream->writeSize(m_lobes.size());
        for (size_t i=0; i<m_lobes.size(); ++i)
            stream->writeFloat(m_lobes[i]);
        stream->writeFloat(m_pitch);
        stream->writeFloat(m_q);
        stream->writeFloat(m_gratingDirU);
        stream->writeBool(m_gratingDirU);
    }

    void configure() {
        unsigned int extraFlags = 0;
        if (!m_specularReflectance->isConstant())
            extraFlags |= ESpatiallyVarying;

        m_components.clear();
        m_components.push_back(EDirectReflection | EFrontSide | extraFlags);
        m_components.push_back(EScatteredReflection | EFrontSide | EUsesSampler | extraFlags);

        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);
        
        m_lobePhis.clear();
        m_lobeEff.clear();
        m_lobePhis.reserve(2*m_lobes.size()*SPECTRUM_SAMPLES);
        for (auto l=0ull;l<m_lobes.size();++l) {
            if (m_lobes[l]<=0) continue;

            for (int i=0;i<SPECTRUM_SAMPLES;++i) {
                const auto phi = (l+1) * Spectrum::lambdas()[i] / m_pitch;
                m_lobePhis.emplace_back(+phi);
                m_lobePhis.emplace_back(-phi);
                m_lobeEff.emplace_back(m_lobes[l]);
            }
        }
        
        if (m_lobeEff.size() == 0)
            Log(EError, "No lobes were supplied!");

        BSDF::configure();
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
            if (name == "specularReflectance")
                m_specularReflectance = static_cast<Texture *>(child);
            else
                BSDF::addChild(name, child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    inline Vector reflect(const Vector &wi) const {
        return Vector(-wi.x, -wi.y, wi.z);
    }
    
    Vector lobeMean(std::size_t comp, const Vector3 &dc_wo, const Vector2 &x) const noexcept {
        const auto phi  = m_lobePhis[comp];
        const auto mean = TQuaternion<Float>::fromAxisAngle(Vector3{ x.x,x.y,0 }, phi)
                        .toTransform()(dc_wo);
        return mean;
    }
    Vector sampleLobe(std::size_t comp, const Vector3 &wi, Float sigma2, 
                      const Vector2 &x, const Point2 &sample) const noexcept {
        const auto k    = lobek(comp);
        const auto theta = warp::squareToStdNormal(sample).x / (k*std::sqrt(sigma2));
        
        const auto &dc_wo = reflect(wi);
        const auto phi  = m_lobePhis[comp];
        return TQuaternion<Float>::fromAxisAngle(Vector3{ x.x,x.y,0 }, phi + theta)
                        .toTransform()(dc_wo);

    }
    Float lobePdf(std::size_t comp, Float sigma2, 
                   const Vector3 &wi, const Vector3 &wo, const Vector2 &x) const noexcept {
        const auto &dc_wo = reflect(wi);
        const auto mean = lobeMean(comp, dc_wo, x);
        const auto k    = lobek(comp);
        
        const auto n = normalize(cross(mean,dc_wo));
        const auto w = normalize(wo - dot(wo,n)*wo);
        const auto theta2 = 1-sqr(dot(w,mean));

        const auto eval = std::exp(-.5f * theta2 * sigma2*sqr(k))
                            * std::sqrt(sigma2/(2*M_PI))*k
                            / (m_lobePhis.size());
        return eval;
    }
    
    const Vector3 gratingXworld(const Intersection &its) const noexcept {
        return normalize(m_gratingDirU ? its.dpdu : its.dpdv);
    }
    const Vector2 gratingX(const Intersection &its) const noexcept {
        const auto &p = gratingXworld(its);
        const auto &v = BSDF::getFrame(its).toLocal(p);
        return { v.x,v.y };
    }
    
    Float lobek(int component) const noexcept {
        const auto bin = (component%(2*SPECTRUM_SAMPLES))/2;
        return Spectrum::ks()[bin];
    }
    auto lobeSpectralBin(int component) const noexcept {
        return (component%(2*SPECTRUM_SAMPLES))/2;
    }
    auto spectrum(int component) const noexcept {
        const auto spectral_bin = lobeSpectralBin(component);
        auto s = Spectrum(.0f);
        s[spectral_bin] = m_lobeEff[component/(2*SPECTRUM_SAMPLES)] * SPECTRUM_SAMPLES;
        return s;
    }
    
    Spectrum envelope(const BSDFSamplingRecord &bRec, Float &eta, EMeasure measure) const {
        const auto hasDC = (bRec.typeMask & EDirectReflection) 
                && (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;
        const auto hasLobes = (bRec.typeMask & EScatteredReflection) 
                && (bRec.component == -1 || bRec.component > 0) && measure == ESolidAngle;
        if ((!hasDC && !hasLobes) 
            || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(.0f);
        
        Assert(!!bRec.pltCtx);
        
        const auto m00 = m_specularReflectance->eval(bRec.its) 
                            * fresnelConductorApprox(Frame::cosTheta(bRec.wi), m_eta, m_k);
        const auto &x = gratingX(bRec.its);
        const auto sigma2 = bRec.pltCtx->sigma2_min_um;

        Spectrum result(.0f);
        if (hasDC) {
            result = m_q * m00;
                    // gaussianSurface::envelopeScattered(m_dcSigma2, *bRec.pltCtx, bRec.wo + bRec.wi);
        } else if (hasLobes) {
            for (auto comp=0;comp<(int)m_lobePhis.size();++comp)
                result += lobePdf(comp, sigma2, bRec.wi, bRec.wo, x) * 
                    (1-m_q) * m00 * spectrum(comp);
        }
        
        return result;
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, Float &eta, 
                  RadiancePacket &rpp, EMeasure measure) const { 
        const auto hasDC = (bRec.typeMask & EDirectReflection) 
                && (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;
        const auto hasLobes = (bRec.typeMask & EScatteredReflection) 
                && (bRec.component == -1 || bRec.component > 0) && measure == ESolidAngle;
        if ((!hasDC && !hasLobes) 
            || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(.0f);
        
        Assert(!!bRec.pltCtx);
        
        const auto m00 = m_specularReflectance->eval(bRec.its);
        const auto &xw = rpp.f.toLocal(gratingXworld(bRec.its));
        const auto &x  = gratingX(bRec.its);

        Spectrum lobes(.0f);
        if (hasLobes) {
            for (auto comp=0;comp<(int)m_lobePhis.size();++comp) {
                const auto spec = spectrum(comp);
                const auto k = Spectrum::ks()[lobeSpectralBin(comp)];

                const auto light_coh_sigma2 = 2*rpp.coherenceSigma2(k, xw, bRec.pltCtx->sigma_zz * 1e+6f);
                lobes += lobePdf(comp, light_coh_sigma2, bRec.wi, bRec.wo, x) * spec;
            }
            lobes.clampNegative();
        }
        const Float DC = hasDC ? 1. : .0;
        
        // Rotate to sp-frame first
        rpp.rotateFrame(bRec.its, Frame::spframe(bRec.wo));
        
        const auto in = rpp.spectrum();
        Spectrum result = Spectrum(.0f);
        for (std::size_t idx=0; idx<rpp.size(); ++idx) {
            // Mueller Fresnel pBSDF
            const auto M = MuellerFresnelRConductor(Frame::cosTheta(bRec.wi), std::complex<Float>(m_eta[idx],m_k[idx]));

            const auto L = m00[idx] * (DC*m_q + (1-m_q)*lobes[idx]) 
                            * ((Matrix4x4)M * rpp.S(idx));
            if (in[idx]>RCPOVERFLOW)
                result[idx] = L[0] / in[idx];
            rpp.L(idx) = L;
        }
        
        return result;
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        const auto hasDC = (bRec.typeMask & EDirectReflection) 
                && (bRec.component == -1 || bRec.component == 0);
        const auto hasLobes = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component > 0);
        if ((!hasDC && !hasLobes)  
            || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return .0f;
        
        Assert(!!bRec.pltCtx);

        Float probDC = hasDC ? 1.0f : 0.0f;
        if (hasDC && hasLobes)
            probDC = m_q;

        if (hasDC && measure == EDiscrete) {
            // pdf += gaussianSurface::scatteredPdf(m_dcSigma2, *bRec.pltCtx, bRec.wi, bRec.wo) * probDC;
            if (std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
                return probDC;
        } else if (hasLobes && measure == ESolidAngle) {
            const auto &x  = gratingX(bRec.its);
            const auto sigma2 = bRec.pltCtx->sigma2_min_um;
            
            Float pdf{};
            for (auto comp=0;comp<(int)m_lobePhis.size();++comp) {
                pdf += lobePdf(comp, sigma2, bRec.wi, bRec.wo, x);
            }

            return pdf;
        }
        
        return 0;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        const auto hasDC = (bRec.typeMask & EDirectReflection) 
                && (bRec.component == -1 || bRec.component == 0);
        const auto hasLobes = (bRec.typeMask & EScatteredReflection) 
                && (bRec.component == -1 || bRec.component > 0);
        if ((!hasDC && !hasLobes) || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(.0f);
        
        Assert(!!bRec.pltCtx);
        
        const auto m00 = m_specularReflectance->eval(bRec.its) 
                            * fresnelConductorApprox(Frame::cosTheta(bRec.wi), m_eta, m_k);
        const auto &x  = gratingX(bRec.its);
        const auto lobeIdx = std::min((int)(sample.y*m_lobePhis.size()),(int)m_lobePhis.size()-1);
        const auto sigma2 = bRec.pltCtx->sigma2_min_um;

        bRec.eta = 1.0f;
        if (hasDC && hasLobes) {
            if (sample.x < m_q) {
                bRec.sampledComponent = 0;
                bRec.sampledType = EDirectReflection;
                bRec.wo = reflect(bRec.wi);
                // bRec.wo = gaussianSurface::sampleScattered(m_dcSigma2, *bRec.pltCtx, bRec.wi, *bRec.sampler);
                
                // const auto pdf = m_q * gaussianSurface::scatteredPdf(m_dcSigma2, *bRec.pltCtx, bRec.wi, bRec.wo);
                // if (pdf <= RCPOVERFLOW)
                //     return Spectrum(.0f);
                return m00;
            } else {
                bRec.sampledComponent = 1;
                bRec.sampledType = EScatteredReflection;
                bRec.wo = sampleLobe(lobeIdx, bRec.wi, sigma2, x, bRec.sampler->next2D());
                if (bRec.wo.z<=0)
                    return Spectrum(.0f);

                return m00;
            }
        } else if (hasDC) {
            bRec.sampledComponent = 0;
            bRec.sampledType = EDirectReflection;
            bRec.wo = reflect(bRec.wi);
            // bRec.wo = gaussianSurface::sampleScattered(m_dcSigma2, *bRec.pltCtx, bRec.wi, *bRec.sampler);
            
            // const auto pdf = gaussianSurface::scatteredPdf(m_dcSigma2, *bRec.pltCtx, bRec.wi, bRec.wo);
            // if (pdf <= RCPOVERFLOW)
            //     return Spectrum(.0f);
            return m_q * m00;
        } else {
            bRec.sampledComponent = 1;
            bRec.sampledType = EScatteredReflection;
            bRec.wo = sampleLobe(lobeIdx, bRec.wi, sigma2, x, bRec.sampler->next2D());
            if (bRec.wo.z<=0)
                return Spectrum(.0f);

            return (1-m_q) * m00;
        }
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "Grating[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  q = " << indent(std::to_string(m_q)) << "," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "  eta = " << m_eta.toString() << "," << endl
            << "  k = " << m_k.toString() << endl
            << "  grating pitch = " << std::to_string(m_pitch) << endl
            << "  grating lobes count = " << std::to_string(m_lobes.size()) << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    std::vector<Float> m_lobes;
    std::vector<Float> m_lobeEff;
    std::vector<Float> m_lobePhis;
    std::vector<std::pair<Float,Float>> m_lobeRange;

    Float m_pitch,m_q;
    bool m_gratingDirU{};

    ref<Texture> m_specularReflectance;
    Spectrum m_eta, m_k;
};


MTS_IMPLEMENT_CLASS_S(Grating, false, BSDF)
MTS_EXPORT_PLUGIN(Grating, "Rough grated surface BRDF");
MTS_NAMESPACE_END
