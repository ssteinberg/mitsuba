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

#include <boost/math/distributions/normal.hpp>

#include "ior.h"

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
        m_q = props.getFloat("q");
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
    }

    void configure() {
        unsigned int extraFlags = 0;
        if (!m_specularReflectance->isConstant())
            extraFlags |= ESpatiallyVarying;

        m_components.clear();
        // DC (specular) lobe
        m_components.push_back(EDirectReflection | EFrontSide | extraFlags);
        // Add a component for each spectral sample per each lobe
        for (auto l=0ull;l<m_lobes.size();++l)
        for (int i=0;i<SPECTRUM_SAMPLES;++i) {
            m_components.push_back(EDirectReflection | EFrontSide | extraFlags);
            m_components.push_back(EDirectReflection | EFrontSide | extraFlags);
        }

        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);
        
        if (m_lobes.size() == 0)
            Log(EError, "No lobes were supplied!");
        
        m_lobePhis.clear();
        m_lobePhis.reserve(m_lobes.size()*SPECTRUM_SAMPLES);
        for (auto l=0ull;l<m_lobes.size();++l)
        for (int i=0;i<SPECTRUM_SAMPLES;++i)
            m_lobePhis.emplace_back(std::asin((l+1) * Spectrum::lambdas()[i] / m_pitch));

        m_norm = {};
        for (auto l=0ull;l<m_lobes.size();++l)
            m_norm += 2*m_lobes[l];
        m_norm = 1/m_norm;

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
    
    Vector sampleLobe(int lobe, Float sigma2, Float k,
                      const Vector3 &wi, const Vector2 &x, Sampler &sampler) const noexcept {
        const auto &dc_wo = reflect(wi);
        if (lobe == 0)
            return dc_wo;
        
        const auto idx  = std::abs(lobe)-1;
        const auto phi  = m_lobePhis[idx] * (lobe>0?1:-1);
        const auto mean = TQuaternion<Float>::fromAxisAngle(Vector3{ x.x,x.y,0 }, phi).toTransform()(dc_wo);

        const auto stddev = Float(1) / std::sqrt(sigma2) / k;
        return warp::squareToTruncatedGaussian(stddev, { mean.x,mean.y }, sampler);
    }
    Float lobePdf(int lobe, Float sigma2, Float k,
                  const Vector3 &wi, const Vector3 &wo, const Vector2 &x) const noexcept {
        if (lobe == 0)
            return 1;
        
        const auto &dc_wo = reflect(wi);
        const auto idx  = std::abs(lobe)-1;
        const auto phi  = m_lobePhis[idx] * (lobe>0?1:-1);
        const auto mean = TQuaternion<Float>::fromAxisAngle(Vector3{ x.x,x.y,0 }, phi).toTransform()(dc_wo);

        const auto stddev = Float(1) / std::sqrt(sigma2) / k;
        return warp::squareToTruncatedGaussianPdf(stddev, { mean.x,mean.y }, wo);
    }
    Float evalLobe(int lobe, Float sigma2,
                   const Vector3 &wi, const Vector3 &wo, const Vector2 &x) const noexcept {
        const auto &dc_wo = reflect(wi);
        if (lobe == 0)
            return std::abs(dot(dc_wo, wo)-1) < DeltaEpsilon ? 1 : 0;
        
        const auto idx  = std::abs(lobe)-1;
        const auto phi  = m_lobePhis[idx] * (lobe>0?1:-1);
        const auto mean = TQuaternion<Float>::fromAxisAngle(Vector3{ x.x,x.y,0 }, phi).toTransform()(dc_wo);
        
        const auto angle_from_mean2 = 1-sqr(dot(mean,wo));  // sin x \approx x, for small x

        // const auto dist = boost::math::normal{ Float(0), Float(1)/sigma2 };
        return std::exp(-.5f*angle_from_mean2*sigma2);//boost::math::pdf(dist, angle_from_mean);
    }
    
    const Vector3 gratingXworld(const Intersection &its) const noexcept {
        return normalize(its.dpdu);
    }
    const Vector2 gratingX(const Intersection &its) const noexcept {
        const auto &p = gratingXworld(its);
        const auto &v = BSDF::getFrame(its).toLocal(p);
        return { v.x,v.y };
    }
    
    auto whichLobe(int component) const noexcept {
        int lobe = (component-1)/(2*SPECTRUM_SAMPLES) + 1;
        lobe *= ((component-1)%2) == 0 ? 1 : -1;
        return lobe;
    }
    auto lobek(int component) const noexcept {
        const auto bin = ((component-1)%(2*SPECTRUM_SAMPLES))/2;
        return Spectrum::ks()[bin];
    }
    auto spectrum(int component, Float &lambda) const noexcept {
        const auto bin = ((component-1)%(2*SPECTRUM_SAMPLES))/2;
        lambda = Spectrum::lambdas()[bin];

        auto s = Spectrum(.0f);
        s[bin] = sqr(Spectrum::lambdas().max()/lambda) * m_lobes[(component-1)/(2*SPECTRUM_SAMPLES)] * m_norm;
        return s;
    }
    auto spectrum(int component) const noexcept {
        Float unused;
        return spectrum(component,unused);
    }

    // Spectrum diffract(const Vector3 &wi, const Vector3 &wo, const Intersection &its, 
    //                   const RadiancePacket &rpp, const PLTContext &pltCtx,
    //                   const Vector3 &local_tangent) const {
    //     const auto pitch = m_pitch;
    //     const auto Phi = -wi+wo;
    //     const auto& PhiPerp = Vector3{ Phi.x * local_tangent.x, Phi.y * local_tangent.y, .0f };
    //     const auto& ppworld = BSDF::getFrame(its).toWorld(PhiPerp);
 
    //     Spectrum ret = Spectrum(.0f);
    //     Float norm{};
    //     for (size_t l=0; l<m_lobes.size(); ++l)
    //     for (int i=0; i<SPECTRUM_SAMPLES; ++i) {
    //         const auto k = Spectrum::ks()[i];

    //         const auto phi = k*Phi;
    //         auto p1 = fmod(pitch*dot(phi,local_tangent) - 2*M_PI*(l+1), 2*M_PI);
    //         auto p2 = fmod(pitch*dot(phi,local_tangent) + 2*M_PI*(l+1), 2*M_PI);
    //         if (p1<-M_PI) p1+=2*M_PI;
    //         if (p2<-M_PI) p2+=2*M_PI;
    //         const auto eff = sqr(2 * INV_PI / Float(l+1)); // Lobe diffraction efficiency

    //         // Convolve the Dirac lobs with the light's coherence
    //         const auto cs = rpp.coherenceLength(k, normalize(rpp.f.toLocal(ppworld)),
    //                                             pltCtx.sigma_zz * 1e+6f);   // in micron
    //         const auto dist = boost::math::normal{ Float(0), Float(1)/cs };
    //         ret[i] += m_lobes[l] * eff * 
    //                  (boost::math::pdf(dist, p1) + 
    //                   boost::math::pdf(dist, p2));
    //         norm += m_lobes[l] * eff;
    //     }

    //     return ret / norm;
    // }
    
    Spectrum envelope(const BSDFSamplingRecord &bRec, Float &eta, EMeasure measure) const {
        const auto hasDirect = (bRec.typeMask & EDirectReflection) && measure == EDiscrete;
        if (!hasDirect || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(.0f);
        
        Assert(!!bRec.pltCtx);
        
        const auto m00 = m_specularReflectance->eval(bRec.its);
        const auto &x = gratingX(bRec.its);
        const auto sigma2 = bRec.pltCtx->sigma_zz * 1e+6f;

        Spectrum result(.0f);
        // DC lobe
        if (bRec.component == -1 || bRec.component == 0) {
            result = evalLobe(0, 0, bRec.wi, bRec.wo, x) * m_q * m00;
        }
        // Diffraction lobes
        if (bRec.component != 0) {
            if (bRec.component != -1) 
                result += evalLobe(whichLobe(bRec.component), sigma2, bRec.wi, bRec.wo, x) * 
                    (1-m_q) * m00 * spectrum(bRec.component);
            else {
                for (auto comp=1;comp<(int)m_components.size();++comp)
                    result += evalLobe(whichLobe(comp), sigma2, bRec.wi, bRec.wo, x) * 
                        (1-m_q) * m00 * spectrum(comp);
            }
        }
        
        return result;
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, Float &eta, 
                  RadiancePacket &rpp, EMeasure measure) const { 
        const auto hasDirect = (bRec.typeMask & EDirectReflection) && measure == EDiscrete;
        if (!hasDirect || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(.0f);
        
        Assert(!!bRec.pltCtx);
        
        const auto m00 = m_specularReflectance->eval(bRec.its);
        const auto &xw = gratingXworld(bRec.its);
        const auto &x  = gratingX(bRec.its);

        Spectrum lobes(.0f);
        // DC lobe
        if (bRec.component == -1 || bRec.component == 0) {
            lobes = evalLobe(0, 0, bRec.wi, bRec.wo, x) * m_q * m00;
        }
        // Diffraction lobes
        if (bRec.component != 0) {
            if (bRec.component != -1) {
                Float lambda;
                const auto spec = spectrum(bRec.component, lambda);
                const auto light_coh_sigma2 = rpp.coherenceSigma2(2*M_PI/lambda, xw, bRec.pltCtx->sigma_zz);
                lobes += evalLobe(whichLobe(bRec.component), light_coh_sigma2, bRec.wi, bRec.wo, x) * 
                    (1-m_q) * m00 * spec;
            }
            else {
                for (auto comp=1;comp<(int)m_components.size();++comp) {
                    Float lambda;
                    const auto spec = spectrum(comp, lambda);
                    const auto light_coh_sigma2 = rpp.coherenceSigma2(2*M_PI/lambda, xw, bRec.pltCtx->sigma_zz);
                    lobes += evalLobe(whichLobe(comp), light_coh_sigma2, bRec.wi, bRec.wo, x) * 
                        (1-m_q) * m00 * spec;
                }
            }
        }

        rpp.rotateFrame(bRec.its, Frame::spframe(bRec.wo));
        
        Spectrum result(.0f);
        const auto in = rpp.spectrum();
        for (std::size_t idx=0; idx<rpp.size(); ++idx) {
            const auto M = MuellerFresnelRConductor(Frame::cosTheta(bRec.wi), std::complex<Float>(m_eta[idx],m_k[idx]));
            auto L = lobes[idx] * ((Matrix4x4)M * rpp.S(idx));
            if (in[idx]>RCPOVERFLOW)
                result[idx] = L[0] / in[idx];
            rpp.L(idx) = L;
        }
        
        return result;
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        const auto hasDirect = (bRec.typeMask & EDirectReflection) && measure == EDiscrete;
        if (!hasDirect || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return .0f;
        
        Assert(!!bRec.pltCtx);
        
        const auto &x = gratingX(bRec.its);
        const auto sigma2 = bRec.pltCtx->sigma_zz * 1e+6f;  // to micron

        Float result = .0f;
        // DC lobe
        if (bRec.component == -1 || bRec.component == 0) {
            result = lobePdf(0, 0, 0, bRec.wi, bRec.wo, x) 
                        * (bRec.component != -1 ? 1.f : m_q);
        }
        // Diffraction lobes
        if (bRec.component != 0) {
            if (bRec.component != -1) 
                result += lobePdf(whichLobe(bRec.component), sigma2, lobek(bRec.component), bRec.wi, bRec.wo, x)
                            * (bRec.component != -1 ? 1.f : 1-m_q);
            else {
                for (auto comp=1;comp<(int)m_components.size();++comp)
                    result += lobePdf(whichLobe(comp), sigma2, lobek(comp), bRec.wi, bRec.wo, x) 
                                * (1-m_q) / Float(m_components.size()-1);
            }
        }
        
        return result;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        const auto hasDirect = !!(bRec.typeMask & EDirectReflection);
        if (!hasDirect || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(.0f);
        
        Assert(!!bRec.pltCtx);
        
        const auto m00 = m_specularReflectance->eval(bRec.its);
        const auto sigma2 = bRec.pltCtx->sigma_zz * 1e+6f;

        auto pdf = 1.f;
        auto weight = m00;
        bool isDC = bRec.component <= 0;
        if (bRec.component == -1) {
            isDC = sample.x <= m_q;
            pdf *= isDC ? m_q : 1.f-m_q;
        }

        Vector3 wo;
        int comp = bRec.component;
        if (isDC) {
            comp = 0;
            weight *= m_q;
            wo = reflect(bRec.wi);
        } else {
            const auto &x  = gratingX(bRec.its);
        
            if (bRec.component == -1) {
                comp = std::min(int(sample.y*m_components.size()-1),(int)m_components.size()-2)+1; 
                pdf  /= m_components.size()-1;
            }
            weight *= spectrum(comp) * (1-m_q);
            wo = sampleLobe(whichLobe(comp), sigma2, lobek(comp), bRec.wi, x, *bRec.sampler);
        }
        
        if (wo.z == 0 || pdf <= RCPOVERFLOW)
            return Spectrum(.0f);
        
        bRec.eta = 1.0f;
        bRec.sampledComponent = comp;
        bRec.sampledType = EDirectReflection;
        bRec.wo = wo;

        return weight / pdf;
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
    std::vector<Float> m_lobePhis;
    Float m_pitch,m_norm,m_q;

    ref<Texture> m_specularReflectance;
    Spectrum m_eta, m_k;
};


MTS_IMPLEMENT_CLASS_S(Grating, false, BSDF)
MTS_EXPORT_PLUGIN(Grating, "Rough grated surface BRDF");
MTS_NAMESPACE_END
