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

#include "ior.h"
#include "mitsuba/render/common.h"

MTS_NAMESPACE_BEGIN

class Grating : public BSDF {
public:
    Grating(const Properties &props) : BSDF(props) {
        ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();

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
        m_gratingDirU = props.getBoolean("gratingInDirectionU", true);

        m_q = props.getFloat("q",0.5);
        m_lobesSamplingWeight = props.getFloat("lobesSamplingWeight",0.9); 
    }

    Grating(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        m_lobes.resize(stream->readSize());
        for (size_t i=0; i<m_lobes.size(); ++i)
            m_lobes[i] = stream->readFloat();
        m_pitch = stream->readFloat();
        m_q = stream->readFloat();
        m_gratingDirU = stream->readBool();
        m_nested = static_cast<BSDF *>(manager->getInstance(stream));

        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        stream->writeSize(m_lobes.size());
        for (size_t i=0; i<m_lobes.size(); ++i)
            stream->writeFloat(m_lobes[i]);
        stream->writeFloat(m_pitch);
        stream->writeFloat(m_q);
        stream->writeFloat(m_gratingDirU);
        manager->serialize(stream, m_nested.get());
    }

    void configure() {
        if (!m_nested)
            Log(EError, "A child BSDF instance is required");

        m_components.clear();
        m_components.push_back(EScatteredReflection | EFrontSide | EUsesSampler);
        for (int i=0; i<m_nested->getComponentCount(); ++i)
            m_components.push_back(m_nested->getType(i));
        
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
        if (child->getClass()->derivesFrom(MTS_CLASS(BSDF))) {
            if (m_nested != NULL)
                Log(EError, "Only a single nested BRDF can be added!");
            m_nested = static_cast<BSDF *>(child);
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
        const auto rcp_stddev = std::sqrt(sigma2)*k;
        
        const auto n = normalize(cross(mean,dc_wo));
        const auto w = normalize(wo - dot(wo,n)*wo);
        const auto theta2 = (1-sqr(dot(w,mean))) * sqr(rcp_stddev);
        
        const auto cdfmin = math::normalCDF((-1 - 0) * rcp_stddev);
        const auto cdfmax = math::normalCDF(( 1 - 0) * rcp_stddev);

        const auto r = (cdfmax-cdfmin) / Float(m_lobePhis.size());
        const auto eval = r>RCPOVERFLOW ? 
                std::exp(-theta2/2) * (INV_TWOPI * rcp_stddev / r)
                : .0f;
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
        s[spectral_bin] = m_lobeEff[component/2];
        return s;
    }
    
    Spectrum envelope(const BSDFSamplingRecord &bRec, Float &eta, EMeasure measure) const {
        const auto hasLobes = (bRec.typeMask & EScatteredReflection) 
                && (bRec.component == -1 || bRec.component == 0) && measure == ESolidAngle;
        bool sampleNested = (bRec.typeMask & m_nested->getType() & BSDF::EAll)
            && (bRec.component == -1 || bRec.component < (int) m_components.size()-1);
        if ((!sampleNested && !hasLobes) 
            || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(.0f);
        
        Assert(!!bRec.pltCtx);
        
        Spectrum result(.0f);
        if (sampleNested) {
            result = m_q * m_nested->envelope(bRec,eta,measure);
        }
        if (hasLobes) {
            Spectrum n,k;
            m_nested->getRefractiveIndex(bRec.its, n, k);
            const auto m00 =  m_nested->getSpecularReflectance(bRec.its)
                                * fresnelConductorApprox(Frame::cosTheta(bRec.wi), n, k);

            const auto &x = gratingX(bRec.its);
            const auto sigma2 = bRec.pltCtx->sigma2_min_um;
            
            for (auto comp=0;comp<(int)m_lobePhis.size();++comp)
                result += lobePdf(comp, sigma2, bRec.wi, bRec.wo, x) * 
                    m00 * spectrum(comp);
        }
        
        return result;
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, Float &eta, 
                  RadiancePacket &rpp, EMeasure measure) const { 
        const auto hasLobes = (bRec.typeMask & EScatteredReflection) 
                && (bRec.component == -1 || bRec.component == 0) && measure == ESolidAngle;
        bool sampleNested = (bRec.typeMask & m_nested->getType() & BSDF::EAll)
            && (bRec.component == -1 || bRec.component < (int) m_components.size()-1);
        if ((!sampleNested && !hasLobes) 
            || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(.0f);
        
        Assert(!!bRec.pltCtx);
        
        auto result = Spectrum(.0f);
        if (sampleNested) {
            result = m_q * m_nested->eval(bRec, rpp, measure);
        }
        if (hasLobes) {
            Spectrum n,k;
            m_nested->getRefractiveIndex(bRec.its, n, k);
            const auto m00 =  m_nested->getSpecularReflectance(bRec.its);
            const auto rcp_max_lambda = 1.f/Spectrum::lambdas().max();

            const auto &xw = rpp.f.toLocal(gratingXworld(bRec.its));
            const auto &x  = gratingX(bRec.its);
                
            Spectrum lobes(.0f);
            for (auto comp=0;comp<(int)m_lobePhis.size();++comp) {
                const auto spec = spectrum(comp);
                const auto k = Spectrum::ks()[lobeSpectralBin(comp)];
                const auto lambda = Spectrum::lambdas()[lobeSpectralBin(comp)] * rcp_max_lambda;

                const auto light_coh_sigma2 = rpp.coherenceSigma2(k, xw, bRec.pltCtx->sigma_zz * 1e+6f);
                lobes += lobePdf(comp, light_coh_sigma2, bRec.wi, bRec.wo, x) * sqr(1/lambda) * spec;
            }
            lobes.clampNegative();

            result += m00 * lobes;
        
            for (std::size_t idx=0; idx<rpp.size(); ++idx) {
                rpp.setL(idx, result[idx]);
            }
        }
        
        return result;
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        const auto hasLobes = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component == 0);
        const auto sampleNested = (bRec.typeMask & m_nested->getType() & BSDF::EAll)
            && (bRec.component == -1 || bRec.component < (int) m_components.size()-1);
        if ((!sampleNested && !hasLobes)  
            || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return .0f;
        
        Assert(!!bRec.pltCtx);

        Float probDC = sampleNested ? 1.0f : 0.0f;
        if (sampleNested && hasLobes)
            probDC = 1-m_lobesSamplingWeight;

        Float pdf{};
        if (hasLobes && measure == ESolidAngle) {
            const auto &x  = gratingX(bRec.its);
            const auto sigma2 = bRec.pltCtx->sigma2_min_um;
            
            for (auto comp=0;comp<(int)m_lobePhis.size();++comp)
                pdf += lobePdf(comp, sigma2, bRec.wi, bRec.wo, x);
            pdf *= (1-probDC);
        }
        if (sampleNested) {
            pdf += probDC * m_nested->pdf(bRec, measure);
        }
        
        return pdf;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample_) const {
        const auto hasLobes = (bRec.typeMask & EScatteredReflection) 
                && (bRec.component == -1 || bRec.component == 0);
        const auto sampleNested = (bRec.typeMask & m_nested->getType() & BSDF::EAll)
            && (bRec.component == -1 || bRec.component < (int) m_components.size()-1);
        if ((!sampleNested && !hasLobes) || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(.0f);
        
        Assert(!!bRec.pltCtx);
        
        auto sample = sample_;
        bool DC = sampleNested;
        Float probDC = sampleNested ? 1 : 0;
        if (sampleNested && hasLobes) {
            probDC = 1-m_lobesSamplingWeight;
            if (sample.x < probDC) {
                sample.x /= probDC;
            } else {
                sample.x = (sample.x - probDC) / (1 - probDC);
                DC = false;
            }
        }
        
        bRec.eta = 1.0f;
        if (DC) {
            const auto pdf = hasLobes ? probDC : 1.0f;
            return m_q * m_nested->sample(bRec, sample) / pdf;
        } else {
            const auto lobeIdx = std::min((int)(sample_.y*m_lobePhis.size()),(int)m_lobePhis.size()-1);
        
            Spectrum n,k;
            m_nested->getRefractiveIndex(bRec.its, n, k);
            const auto m00 =  m_nested->getSpecularReflectance(bRec.its)
                                * fresnelConductorApprox(Frame::cosTheta(bRec.wi), n, k);
            
            const auto &x  = gratingX(bRec.its);
            const auto sigma2 = bRec.pltCtx->sigma2_min_um;

            bRec.sampledComponent = 0;
            bRec.sampledType = EScatteredReflection;
            bRec.wo = sampleLobe(lobeIdx, bRec.wi, sigma2, x, bRec.sampler->next2D());
            if (bRec.wo.z<=0)
                return Spectrum(.0f);

            const auto pdf = sampleNested ? 1-probDC : 1.0f;
            auto result = Spectrum(.0f);
            for (auto comp=0;comp<(int)m_lobePhis.size();++comp)
                result += lobePdf(comp, sigma2, bRec.wi, bRec.wo, x) * 
                    m00 * spectrum(comp);

            return result / pdf;
        }
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "Grating[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  q = " << indent(std::to_string(m_q)) << "," << endl
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

    Float m_pitch,m_q;
    Float m_lobesSamplingWeight{ .9 };
    bool m_gratingDirU{};
    
    ref<BSDF> m_nested;
};


MTS_IMPLEMENT_CLASS_S(Grating, false, BSDF)
MTS_EXPORT_PLUGIN(Grating, "Diffraction grating BRDF");
MTS_NAMESPACE_END
