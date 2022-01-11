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
#include <mitsuba/plt/gaussianSurface.hpp>
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

        m_q = new ConstantFloatTexture(props.getFloat("q", .0f));

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
        m_specularSamplingWeight = .1f;
    }

    Grating(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        m_q = static_cast<Texture *>(manager->getInstance(stream));
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_eta = Spectrum(stream);
        m_k = Spectrum(stream);
        
        m_lobes.resize(stream->readSize());
        for (size_t i=0; i<m_lobes.size(); ++i)
            m_lobes[i] = stream->readFloat();
        m_pitch = stream->readFloat();

        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        manager->serialize(stream, m_q.get());
        manager->serialize(stream, m_specularReflectance.get());
        m_eta.serialize(stream);
        m_k.serialize(stream);

        stream->writeSize(m_lobes.size());
        for (size_t i=0; i<m_lobes.size(); ++i)
            stream->writeFloat(m_lobes[i]);
        stream->writeFloat(m_pitch);
    }

    void configure() {
        unsigned int extraFlags = 0;
        unsigned int extraFlagsScattered = 0;
        if (!m_q->isConstant() || !m_specularReflectance->isConstant())
            extraFlags |= ESpatiallyVarying;

        m_components.clear();
        m_components.push_back(EDirectReflection | EFrontSide | extraFlags);
        m_components.push_back(EScatteredReflection | EFrontSide | EUsesSampler | extraFlags | extraFlagsScattered);

        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);
        
        if (m_lobes.size() == 0)
            Log(EError, "No lobes were supplied!");
        
        // Poorly approximated fit to some grating geometries, should be done much better.
        m_sigma2 = m_pitch/10.f;

        BSDF::configure();
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
            if (name == "q")
                m_q = static_cast<Texture *>(child);
            else if (name == "specularReflectance")
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

    Spectrum diffract(const Vector3 &wi, const Vector3 &wo, const Intersection &its, 
                      const RadiancePacket &rpp, const PLTContext &pltCtx,
                      const Vector3 &local_tangent) const {
        const auto pitch = m_pitch;
        const auto Phi = -wi+wo;
        const auto& PhiPerp = Vector3{ Phi.x * local_tangent.x, Phi.y * local_tangent.y, .0f };
        const auto& ppworld = BSDF::getFrame(its).toWorld(PhiPerp);
 
        Spectrum ret = Spectrum(.0f);
        Float norm{};
        for (size_t l=0; l<m_lobes.size(); ++l)
        for (int i=0; i<SPECTRUM_SAMPLES; ++i) {
            const auto k = Spectrum::ks()[i];

            const auto phi = k*Phi;
            auto p1 = fmod(pitch*dot(phi,local_tangent) - 2*M_PI*(l+1), 2*M_PI);
            auto p2 = fmod(pitch*dot(phi,local_tangent) + 2*M_PI*(l+1), 2*M_PI);
            if (p1<-M_PI) p1+=2*M_PI;
            if (p2<-M_PI) p2+=2*M_PI;
            const auto eff = sqr(2 * INV_PI / Float(l+1)); // Lobe diffraction efficiency

            // Convolve the Dirac lobs with the light's coherence
            const auto cs = rpp.coherenceLength(k, rpp.f.toLocal(ppworld), 
                                                pltCtx.sigma_zz * 1e+6f);   // in micron
            const auto dist = boost::math::normal{ Float(0), cs };
            ret[i] += m_lobes[l] * eff * 
                     (boost::math::pdf(dist, p1) + 
                      boost::math::pdf(dist, p2));
            norm += m_lobes[l] * eff;
        }

        return ret / norm;
    }
    
    Spectrum envelope(const BSDFSamplingRecord &bRec, Float &eta, EMeasure measure) const {
        const auto hasDirect = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0)
                && measure == EDiscrete;
        const auto hasScattered = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component == 1)
                && measure == ESolidAngle;

        if ((!hasDirect && !hasScattered)
            || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(0.0f);
        
        Assert(!!bRec.pltCtx);
        
        const auto m00 = m_specularReflectance->eval(bRec.its);
        const auto q = m_q->eval(bRec.its).average();
        const auto a = gaussianSurface::alpha(Frame::cosTheta(bRec.wi), q);
        
        if (!hasDirect && a.average()==1)
            return Spectrum(.0f);
        
        if (hasDirect) {
            if (std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
                return a * m00 *
                    fresnelConductorApprox(Frame::cosTheta(bRec.wi), m_eta, m_k);
        }
        if (hasScattered) {
            const auto m = normalize(bRec.wo+bRec.wi);
            return (Spectrum(1.f)-a) * m00 * 
                    gaussianSurface::envelopeScattered(m_sigma2, *bRec.pltCtx, bRec.wo + bRec.wi) *
                    fresnelConductorApprox(dot(m,bRec.wi), m_eta, m_k);
        }
        return Spectrum(.0f);
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, Float &eta, 
                  RadiancePacket &rpp, EMeasure measure) const { 
        const auto hasDirect = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0)
                && measure == EDiscrete;
        const auto hasScattered = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component == 1)
                && measure == ESolidAngle;

        if ((!hasDirect && !hasScattered)
            || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(0.0f);
        
        Assert(bRec.mode==EImportance && rpp.isValid());
        Assert(!!bRec.pltCtx);
        
        const auto m00 = m_specularReflectance->eval(bRec.its);
        const auto q = m_q->eval(bRec.its).average();
        const auto a = gaussianSurface::alpha(Frame::cosTheta(bRec.wi), q);
        const auto m = normalize(bRec.wo+bRec.wi);
        const auto rcp_max_lambda = 1.f/Spectrum::lambdas().max();
        
        if (!hasDirect && a.average() >= 1-Epsilon)
            return Spectrum(.0f);
        if (hasDirect && std::abs(dot(reflect(bRec.wi), bRec.wo)-1) >= DeltaEpsilon)
            return Spectrum(.0f);
        
        const auto fi = rpp.f;
        Matrix3x3 Q = Matrix3x3(bRec.its.toLocal(fi.s),
                                bRec.its.toLocal(fi.t),
                                bRec.its.toLocal(fi.n)), Qt;
        Q.transpose(Qt);
        
        // Rotate to sp-frame first
        rpp.rotateFrame(bRec.its, Frame::spframe(bRec.wo));

        // Diffraction
        auto D = Spectrum(1.f);
        if (!hasDirect) {
            const Intersection &its = bRec.its;
            const auto &frame = BSDF::getFrame(its);
            const auto uv = its.uv;
            const auto R  = uv-Point2(.5f,.5f);
            const auto p  = normalize(its.dpdu*R.x + its.dpdv*R.y);
            const auto lt = cross(Vector3(0,0,1),frame.toLocal(p));
            D = diffract(bRec.wi, bRec.wo, its, rpp, *bRec.pltCtx, lt);
        }
        
        const auto in = rpp.spectrum();
        Spectrum result = Spectrum(.0f);
        for (std::size_t idx=0; idx<rpp.size(); ++idx) {
            // Mueller Fresnel pBSDF
            const auto lambda = Spectrum::lambdas()[idx] * rcp_max_lambda;
            const auto M = MuellerFresnelRConductor(Frame::cosTheta(bRec.wi), std::complex<Float>(m_eta[idx],m_k[idx]));

            auto L = m00[idx] * ((Matrix4x4)M * rpp.S(idx));
            if (!hasDirect)
                L *= sqr(1/lambda) * D[idx] * Frame::cosTheta(bRec.wo);

            if (in[idx]>RCPOVERFLOW)
                result[idx] = L[0] / in[idx];
            rpp.L(idx) = L;
        }

        return result;
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        const auto hasDirect = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0);
        const auto hasScattered = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component == 1);

        if (Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return .0f;

        Assert(!!bRec.pltCtx);

        Float probDirect = hasDirect ? 1.0f : 0.0f;
        if (hasDirect && hasScattered) {
            const auto q = m_q->eval(bRec.its).average();
            probDirect = gaussianSurface::alpha(Frame::cosTheta(bRec.wi), q).average();
        }

        if (hasDirect && measure == EDiscrete) {
            /* Check if the provided direction pair matches an ideal
               specular reflection; tolerate some roundoff errors */
            if (std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
                return probDirect;
        } else if (hasScattered && measure == ESolidAngle && probDirect<1) {
            return /* Frame::cosTheta(bRec.wo) */ 
                gaussianSurface::scatteredPdf(m_sigma2, *bRec.pltCtx, bRec.wi, bRec.wo) * (1-probDirect);
        }

        return 0.0f;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        const auto hasDirect = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0);
        const auto hasScattered = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component == 1);

        if ((!hasScattered && !hasDirect) || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(0.0f);

        Assert(!!bRec.pltCtx);
        
        const auto m00 = m_specularReflectance->eval(bRec.its);
        const auto q = m_q->eval(bRec.its).average();
        const auto a = gaussianSurface::alpha(Frame::cosTheta(bRec.wi), q);

        bRec.eta = 1.0f;
        if (hasScattered && hasDirect) {
            const auto probDirect = a.average();

            if (probDirect==1 || sample.x < probDirect) {
                bRec.sampledComponent = 0;
                bRec.sampledType = EDirectReflection;
                bRec.wo = reflect(bRec.wi);

                return 1.f/probDirect * a * m00 *
                    fresnelConductorApprox(Frame::cosTheta(bRec.wi), m_eta, m_k);
            } else {
                bRec.sampledComponent = 1;
                bRec.sampledType = EScatteredReflection;
                bRec.wo = gaussianSurface::sampleScattered(m_sigma2, *bRec.pltCtx, bRec.wi, *bRec.sampler);

                const auto pdf = (1-probDirect) * gaussianSurface::scatteredPdf(m_sigma2, *bRec.pltCtx, bRec.wi, bRec.wo);
                if (pdf <= RCPOVERFLOW)
                    return Spectrum(.0f);

                const auto m = normalize(bRec.wo+bRec.wi);
                return (1.f/pdf) * (Spectrum(1.f)-a) * m00 *
                    fresnelConductorApprox(dot(m,bRec.wi), m_eta, m_k);
            }
        } else if (hasDirect) {
            bRec.sampledComponent = 0;
            bRec.sampledType = EDirectReflection;
            bRec.wo = reflect(bRec.wi);
            return a * m00 *
                fresnelConductorApprox(Frame::cosTheta(bRec.wi), m_eta, m_k);
        } else {
            bRec.sampledComponent = 1;
            bRec.sampledType = EScatteredReflection;
            bRec.wo = gaussianSurface::sampleScattered(m_sigma2, *bRec.pltCtx, bRec.wi, *bRec.sampler);

            const auto pdf = gaussianSurface::scatteredPdf(m_sigma2, *bRec.pltCtx, bRec.wi, bRec.wo);
            if (pdf <= RCPOVERFLOW)
                return Spectrum(.0f);

            const auto m = normalize(bRec.wo+bRec.wi);
            return (1.f/pdf) * (Spectrum(1.f)-a) * m00 *
                fresnelConductorApprox(dot(m,bRec.wi), m_eta, m_k);
        }
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "Grating[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  q = " << indent(m_q->toString()) << "," << endl
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
    Float m_pitch,m_sigma2,m_specularSamplingWeight;

    ref<Texture> m_specularReflectance;
    ref<Texture> m_q;
    Spectrum m_eta, m_k;
};


MTS_IMPLEMENT_CLASS_S(Grating, false, BSDF)
MTS_EXPORT_PLUGIN(Grating, "Rough grated surface BRDF");
MTS_NAMESPACE_END
