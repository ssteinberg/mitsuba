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

#include <mitsuba/core/fresolver.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/plt/mueller.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/shape.h>

#include <mitsuba/plt/gaussianSurface.hpp>

#include "microfacet.h"
#include "ior.h"

MTS_NAMESPACE_BEGIN

class Conductor : public BSDF {
public:
    Conductor(const Properties &props) : BSDF(props) {
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
        m_sigma2 = new ConstantFloatTexture(props.getFloat("sigma2", .0f));
    }

    Conductor(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        m_q = static_cast<Texture *>(manager->getInstance(stream));
        m_sigma2 = static_cast<Texture *>(manager->getInstance(stream));
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_eta = Spectrum(stream);
        m_k = Spectrum(stream);

        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        manager->serialize(stream, m_q.get());
        manager->serialize(stream, m_sigma2.get());
        manager->serialize(stream, m_specularReflectance.get());
        m_eta.serialize(stream);
        m_k.serialize(stream);
    }

    void configure() {
        unsigned int extraFlags = 0;
        unsigned int extraFlagsScattered = 0;
        if (!m_q->isConstant() || !m_specularReflectance->isConstant())
            extraFlags |= ESpatiallyVarying;
        if (!m_sigma2->isConstant())
            extraFlagsScattered |= ESpatiallyVarying;

        m_components.clear();
        m_components.push_back(EDirectReflection | EFrontSide | extraFlags);
        m_components.push_back(EScatteredReflection | EFrontSide | EUsesSampler | extraFlags | extraFlagsScattered);

        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);

        BSDF::configure();
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
            if (name == "q")
                m_q = static_cast<Texture *>(child);
            else if (name == "sigma2")
                m_sigma2 = static_cast<Texture *>(child);
            else if (name == "specularReflectance")
                m_specularReflectance = static_cast<Texture *>(child);
            else
                BSDF::addChild(name, child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    inline Vector reflect(const Vector &wi) const {
        const auto m = Vector3{ 0,0,1 };
        return 2*dot(wi,m)*m - wi;
    }
    
    Spectrum envelope(const BSDFSamplingRecord &bRec, EMeasure measure) const {
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
        const auto sigma2 = m_sigma2->eval(bRec.its).average();
        const auto costheta_o = Frame::cosTheta(bRec.wo);
        const auto a = gaussianSurface::alpha(Frame::cosTheta(bRec.wi), q);
        
        if (hasDirect) {
            if (std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
                return a * m00 *
                    fresnelConductorApprox(Frame::cosTheta(bRec.wi), m_eta, m_k);
            return Spectrum(.0f);
        }
        const auto m = normalize(bRec.wo+bRec.wi);
        return costheta_o * (Spectrum(1.f)-a) * m00 * 
                gaussianSurface::envelopeScattered(sigma2, *bRec.pltCtx, bRec.wo + bRec.wi) *
                fresnelConductorApprox(dot(m,bRec.wi), m_eta, m_k);
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, RadiancePacket &radiancePacket, EMeasure measure) const { 
        const auto hasDirect = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0)
                && measure == EDiscrete;
        const auto hasScattered = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component == 1)
                && measure == ESolidAngle;

        if ((!hasDirect && !hasScattered)
            || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(0.0f);
        
        Assert(bRec.mode==EImportance && radiancePacket.isValid());
        Assert(!!bRec.pltCtx);
        
        const auto m00 = m_specularReflectance->eval(bRec.its);
        const auto q = m_q->eval(bRec.its).average();
        const auto sigma2 = m_sigma2->eval(bRec.its).average();
        const auto costheta_o = Frame::cosTheta(bRec.wo);
        const auto a = gaussianSurface::alpha(Frame::cosTheta(bRec.wi), q);
        const auto m = normalize(bRec.wo+bRec.wi);
        
        const auto fi = radiancePacket.f;
        Matrix3x3 Q = Matrix3x3(bRec.its.toLocal(fi.s),
                                bRec.its.toLocal(fi.t),
                                bRec.its.toLocal(fi.n)), Qt;
        Q.transpose(Qt);
        
        // Rotate to sp-frame first
        radiancePacket.rotateFrame(bRec.its, Frame::spframe(bRec.wo,Normal{ 0,0,1 }));
        
        const auto& in = radiancePacket.spectrum();
        Spectrum result = Spectrum(.0f);
        for (std::size_t idx=0; idx<radiancePacket.size(); ++idx) {
            if (hasDirect) {
                if (std::abs(dot(reflect(bRec.wi), bRec.wo)-1) >= DeltaEpsilon)
                    return Spectrum(.0f);
                
                // Mueller Fresnel pBSDF
                const auto M = MuellerFresnelRConductor(Frame::cosTheta(bRec.wi), std::complex<Float>(m_eta[idx],m_k[idx]));
                radiancePacket.L(idx) = a[idx] * m00[idx] * ((Matrix4x4)M * radiancePacket.S(idx));
            }
            else {
                const auto k = Spectrum::ks()[idx];
                const auto h = k*(bRec.wo + bRec.wi);
                const auto D = gaussianSurface::diffract(radiancePacket.invTheta(k,bRec.pltCtx->sigma_zz * 1e+6f), 
                                        sigma2, Q,Qt, h);
                // const auto Dx = diffract(radiancePacket.invThetax(k,bRec.pltCtx->sigma_zz * 1e+6f), 
                //                          sigma2, Q,Qt, h);
                // const auto Dy = diffract(radiancePacket.invThetay(k,bRec.pltCtx->sigma_zz * 1e+6f), 
                //                          sigma2, Q,Qt, h);
                // const auto Dc = diffract(radiancePacket.invThetac(k,bRec.pltCtx->sigma_zz * 1e+6f), 
                //                          sigma2, Q,Qt, h);
                //                          
                const auto M = MuellerFresnelRConductor(dot(m,bRec.wi), std::complex<Float>(m_eta[idx],m_k[idx]));
                
                radiancePacket.L(idx) = costheta_o * (1-a[idx]) * m00[idx] * ((Matrix4x4)M *
                                D*radiancePacket.S(idx));
                //                (Dx*radiancePacket.Sx(idx) + Dy*radiancePacket.Sy(idx) + Dc*radiancePacket.Sc(idx)));
            }

            if (in[idx]>RCPOVERFLOW)
                result[idx] = radiancePacket.L(idx)[0] / in[idx];
        }

        return result;
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        const auto hasDirect = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0);
        const auto hasScattered = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component == 1);

        if (Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return 0.0f;

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
            const auto sigma2 = m_sigma2->eval(bRec.its).average();
            return /* Frame::cosTheta(bRec.wo) */ 
                gaussianSurface::scatteredPdf(sigma2, *bRec.pltCtx, bRec.wi, bRec.wo) * (1-probDirect);
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
        const auto sigma2 = m_sigma2->eval(bRec.its).average();
        const auto a = gaussianSurface::alpha(Frame::cosTheta(bRec.wi), q);

        bRec.eta = 1.0f;
        if (hasScattered && hasDirect) {
            const auto probDirect = a.average();

            if (probDirect==1 || sample.x < probDirect) {
                bRec.sampledComponent = 0;
                bRec.sampledType = EDirectReflection;
                bRec.wo = reflect(bRec.wi);

                return Float(1)/probDirect * a * m00 *
                    fresnelConductorApprox(Frame::cosTheta(bRec.wi), m_eta, m_k);
            } else {
                bRec.sampledComponent = 1;
                bRec.sampledType = EScatteredReflection;
                bRec.wo = gaussianSurface::sampleScattered(sigma2, *bRec.pltCtx, bRec.wi, *bRec.sampler);

                const auto m = normalize(bRec.wo+bRec.wi);
                return Float(1)/(1-probDirect) * (Spectrum(1.f)-a) * m00 *
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
            bRec.wo = gaussianSurface::sampleScattered(sigma2, *bRec.pltCtx, bRec.wi, *bRec.sampler);

            const auto m = normalize(bRec.wo+bRec.wi);
            return (Spectrum(1.f)-a) * m00 *
                fresnelConductorApprox(dot(m,bRec.wi), m_eta, m_k);
        }
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "Conductor[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  q = " << indent(m_q->toString()) << "," << endl
            << "  sigma2 = " << indent(m_sigma2->toString()) << "," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "  eta = " << m_eta.toString() << "," << endl
            << "  k = " << m_k.toString() << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_specularReflectance;
    ref<Texture> m_q, m_sigma2;
    Spectrum m_eta, m_k;
};


MTS_IMPLEMENT_CLASS_S(Conductor, false, BSDF)
MTS_EXPORT_PLUGIN(Conductor, "Rough conductor BRDF");
MTS_NAMESPACE_END
