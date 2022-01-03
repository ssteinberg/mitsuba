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
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/hw/basicshader.h>
#include "microfacet.h"
#include "ior.h"
#include "mitsuba/core/constants.h"
#include "mitsuba/core/util.h"
#include "mitsuba/plt/mueller.h"
#include "mitsuba/render/common.h"
#include <mitsuba/plt/plt.h>
#include <mitsuba/plt/gaussianSurface.hpp>

MTS_NAMESPACE_BEGIN

class Dielectric : public BSDF {
public:
    Dielectric(const Properties &props) : BSDF(props) {
        m_specularReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("specularReflectance", Spectrum(1.0f)));
        m_specularTransmittance = new ConstantSpectrumTexture(
            props.getSpectrum("specularTransmittance", Spectrum(1.0f)));

        Float intIOR = lookupIOR(props, "intIOR", "bk7");
        Float extIOR = lookupIOR(props, "extIOR", "air");

        if (intIOR < 0 || extIOR < 0 || intIOR == extIOR)
            Log(EError, "The interior and exterior indices of "
                "refraction must be positive and differ!");

        m_eta = intIOR / extIOR;
        m_invEta = 1.f / m_eta;

        m_q = new ConstantFloatTexture(props.getFloat("q", .0f));
        m_sigma2 = new ConstantFloatTexture(props.getFloat("sigma2", .0f));
    }

    Dielectric(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        m_q = static_cast<Texture *>(manager->getInstance(stream));
        m_sigma2 = static_cast<Texture *>(manager->getInstance(stream));
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_specularTransmittance = static_cast<Texture *>(manager->getInstance(stream));
        m_eta = stream->readFloat();
        m_invEta = 1.f / m_eta;

        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        manager->serialize(stream, m_q.get());
        manager->serialize(stream, m_sigma2.get());
        manager->serialize(stream, m_specularReflectance.get());
        manager->serialize(stream, m_specularTransmittance.get());
        stream->writeFloat(m_eta);
    }

    void configure() {
        unsigned int extraFlags = 0;
        unsigned int extraFlagsScattered = 0;
        if (!m_q->isConstant() || !m_specularReflectance->isConstant())
            extraFlags |= ESpatiallyVarying;
        if (!m_sigma2->isConstant())
            extraFlagsScattered |= ESpatiallyVarying;

        m_components.clear();
        m_components.push_back(EDirectReflection | EFrontSide | EBackSide | extraFlags);
        m_components.push_back(EDirectTransmission | EFrontSide | EBackSide | extraFlags);
        m_components.push_back(EScatteredReflection | EFrontSide | EBackSide | 
                               EUsesSampler | extraFlags | extraFlagsScattered);
        m_components.push_back(EScatteredTransmission | EFrontSide | EBackSide | 
                               EUsesSampler | extraFlags | extraFlagsScattered);

        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);
        m_specularTransmittance = ensureEnergyConservation(
            m_specularTransmittance, "specularTransmittance", 1.0f);

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
            else if (name == "specularTransmittance")
                m_specularTransmittance = static_cast<Texture *>(child);
            else
                BSDF::addChild(name, child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    inline Vector reflect(const Vector &wi) const {
        return Vector(-wi.x, -wi.y, wi.z);
    }
    inline Vector refract(const Vector &wi, Float cosThetaT) const {
        Float scale = -(cosThetaT < 0 ? m_invEta : m_eta);
        return Vector(scale*wi.x, scale*wi.y, cosThetaT);
    }
    // Not accurate!!
    inline Vector refract(const Vector &w) const {
        Float scale = -(w.z > .0f ? m_invEta : m_eta);
        auto v = scale * Vector{ w.x, w.y, 0 };
        const auto l = v.lengthSquared();
        if (l>1)
            return Vector{ 0,0,0 };
        v.z = std::sqrt(1-l) * (-math::sgn(w.z));
        return v;
    }
    inline Vector scattered_wo(const Vector &wo) const {
        return -reflect(wo);//reflect(refract(wo));
    }
    
    Spectrum envelope(const BSDFSamplingRecord &bRec, Float &eta, EMeasure measure) const {
        const auto hasDirect = (bRec.typeMask & EDirect)
                && (bRec.component == -1 || bRec.component == 0 || bRec.component == 1)
                && measure == EDiscrete;
        const auto hasScattered = (bRec.typeMask & EScattered)
                && (bRec.component == -1 || bRec.component == 2 || bRec.component == 3)
                && measure == ESolidAngle;
        const auto isReflection = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) > 0;

        if ((!hasDirect && !hasScattered) || Frame::cosTheta(bRec.wi) == 0 ||
            (isReflection && bRec.component != -1 && bRec.component != 0 && bRec.component != 2))
            return Spectrum(0.0f);
        
        Assert(!!bRec.pltCtx);
        
        const auto m00 = isReflection ? m_specularReflectance->eval(bRec.its) : 
                                        m_specularTransmittance->eval(bRec.its);
        const auto q = m_q->eval(bRec.its).average();
        const auto sigma2 = m_sigma2->eval(bRec.its).average();
        const auto a = gaussianSurface::alpha(Frame::cosTheta(bRec.wi), q);

        if (!hasDirect && a.average()==1)
            return Spectrum(.0f);

        const auto costheta_o = std::fabs(Frame::cosTheta(bRec.wo));
        const auto& wo = isReflection ? bRec.wo : scattered_wo(bRec.wo);
        
        if (wo.z == 0)
            return Spectrum(.0f);
        
        eta = isReflection ? 1.0f : bRec.wo.z<.0f ? m_eta : m_invEta;
        if (hasDirect) {
            Float costheta_t;
            const auto Fr = fresnelDielectricExt(Frame::cosTheta(bRec.wi),costheta_t,m_eta);

            if (isReflection  && std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
                return Fr * a * m00;
            if (!isReflection && std::abs(dot(refract(bRec.wi, costheta_t), bRec.wo)-1) < DeltaEpsilon)
                return (1-Fr) * a * m00;
        }
        if (hasScattered) {
            const auto h = bRec.wi + wo;
            const auto m = normalize(h);
            const auto r = fresnelDielectricExt(dot(bRec.wi,m),m_eta);

            return costheta_o * (isReflection ? r : 1.f-r) *
                    (Spectrum(1.f)-a) * m00 * 
                    gaussianSurface::envelopeScattered(sigma2, *bRec.pltCtx, h);
        }
        return Spectrum(.0f);
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, Float &eta,
                  RadiancePacket &radiancePacket, EMeasure measure) const { 
        const auto hasDirect = (bRec.typeMask & EDirect)
                && (bRec.component == -1 || bRec.component == 0 || bRec.component == 1)
                && measure == EDiscrete;
        const auto hasScattered = (bRec.typeMask & EScattered)
                && (bRec.component == -1 || bRec.component == 2 || bRec.component == 3)
                && measure == ESolidAngle;
        const auto isReflection = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) > 0;

        if ((!hasDirect && !hasScattered) || Frame::cosTheta(bRec.wi) == 0 ||
            (isReflection && bRec.component != -1 && bRec.component != 0 && bRec.component != 2))
            return Spectrum(0.0f);
        
        Assert(bRec.mode==EImportance && radiancePacket.isValid());
        Assert(!!bRec.pltCtx);
        
        const auto m00 = isReflection ? m_specularReflectance->eval(bRec.its) : 
                                        m_specularTransmittance->eval(bRec.its);
        const auto q = m_q->eval(bRec.its).average();
        const auto sigma2 = m_sigma2->eval(bRec.its).average();
        const auto a = gaussianSurface::alpha(Frame::cosTheta(bRec.wi), q);

        const auto costheta_o = std::fabs(Frame::cosTheta(bRec.wo));
        const auto& wo = isReflection ? bRec.wo : -reflect(bRec.wo);
        const auto h = bRec.wi + wo;
        const auto m = normalize(h);
        
        if (wo.z == 0)
            return Spectrum(.0f);
        Assert(wo.z*bRec.wi.z>=.0f);
        
        if (!hasDirect && a.average()==1)
            return Spectrum(.0f);
        if (hasDirect) {
            Float costheta_t;
            fresnelDielectricExt(Frame::cosTheta(bRec.wi),costheta_t,m_eta);

            if ((isReflection  && std::abs(dot(reflect(bRec.wi), bRec.wo)-1) >= DeltaEpsilon) ||
                (!isReflection && std::abs(dot(refract(bRec.wi, costheta_t), bRec.wo)-1) >= DeltaEpsilon))
                return Spectrum(.0f);
        }
        
        const auto fi = radiancePacket.f;
        Matrix3x3 Q = Matrix3x3(bRec.its.toLocal(fi.s),
                                bRec.its.toLocal(fi.t),
                                bRec.its.toLocal(fi.n)), Qt;
        Q.transpose(Qt);
        
        // Rotate to sp-frame first
        radiancePacket.rotateFrame(bRec.its, Frame::spframe(bRec.wo));

        const auto k = Spectrum::ks().average();
        const auto D = !hasDirect ? 
                       gaussianSurface::diffract(radiancePacket.invTheta(k,bRec.pltCtx->sigma_zz * 1e+6f), 
                                                 sigma2, Q,Qt, k*h) : .0f;
        const auto M = !hasDirect ? 
                       MuellerFresnelDielectric(dot(bRec.wi,m), m_eta, isReflection) :
                       MuellerFresnelDielectric(Frame::cosTheta(bRec.wi), m_eta, isReflection);
        
        const auto& in = radiancePacket.spectrum();
        Spectrum result = Spectrum(.0f);
        for (std::size_t idx=0; idx<radiancePacket.size(); ++idx) {
            auto L = m00[idx] * ((Matrix4x4)M * radiancePacket.S(idx));
            L *= hasDirect ?
                a[idx] :
                (costheta_o * (1-a[idx]) * D);

            if (in[idx]>RCPOVERFLOW)
                result[idx] = L[0] / in[idx];
            radiancePacket.L(idx) = L;
        }

        eta = isReflection ? 1.0f : bRec.wo.z<.0f ? m_eta : m_invEta;
        Assert(std::isfinite(result[0]));
        return result;
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        const auto hasDirect = (bRec.typeMask & EDirect)
                && (bRec.component == -1 || bRec.component == 0 || bRec.component == 1);
        const auto hasScattered = (bRec.typeMask & EScattered)
                && (bRec.component == -1 || bRec.component == 2 || bRec.component == 3);
        const auto hasReflection = (bRec.typeMask & EReflection)
                && (bRec.component == -1 || bRec.component == 0 || bRec.component == 2);
        const auto hasTransmission = (bRec.typeMask & ETransmission)
                && (bRec.component == -1 || bRec.component == 1 || bRec.component == 3);
        const auto isReflection = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) > 0;

        if (Frame::cosTheta(bRec.wi) == 0 || (!hasReflection && !hasTransmission) || (!hasDirect && !hasScattered) 
                || (isReflection && !hasReflection)|| (!isReflection && !hasTransmission))
            return .0f;

        Assert(!!bRec.pltCtx);

        Float probDirect = hasDirect ? 1.f : .0f;
        if (hasDirect && hasScattered) {
            const auto q = m_q->eval(bRec.its).average();
            probDirect = gaussianSurface::alpha(Frame::cosTheta(bRec.wi), q).average();
        }

        if (hasDirect && measure == EDiscrete) {
            Float costheta_t;
            const auto Fr = fresnelDielectricExt(Frame::cosTheta(bRec.wi),costheta_t,m_eta);

            Float F = 1.f;
            if (hasReflection && hasTransmission)
                F = isReflection ? Fr : 1.f-Fr;

            if ((isReflection  && std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon) ||
                (!isReflection && std::abs(dot(refract(bRec.wi, costheta_t), bRec.wo)-1) < DeltaEpsilon))
                return F * probDirect;
        } else if (hasScattered && measure == ESolidAngle && probDirect<1) {
            const auto& wo = isReflection ? bRec.wo : scattered_wo(bRec.wo);
            if (wo.z == 0)
                return .0f;
            const auto sigma2 = m_sigma2->eval(bRec.its).average();
            const auto h = bRec.wi + wo;
            const auto m = normalize(h);
            
            const auto r = fresnelDielectricExt(dot(bRec.wi,m),m_eta);
            Float F = 1.f;
            if (hasReflection && hasTransmission)
                F = isReflection ? r : 1.f-r;
            
            return gaussianSurface::scatteredPdf(sigma2, *bRec.pltCtx, bRec.wi, wo) * 
                F * (1-probDirect);
        }

        return .0f;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        const auto hasDirect = (bRec.typeMask & EDirect)
                && (bRec.component == -1 || bRec.component == 0 || bRec.component == 1);
        const auto hasScattered = (bRec.typeMask & EScattered)
                && (bRec.component == -1 || bRec.component == 2 || bRec.component == 3);
        const auto hasReflection = (bRec.typeMask & EReflection)
                && (bRec.component == -1 || bRec.component == 0 || bRec.component == 2);
        const auto hasTransmission = (bRec.typeMask & ETransmission)
                && (bRec.component == -1 || bRec.component == 1 || bRec.component == 3);

        if (Frame::cosTheta(bRec.wi) == 0 || (!hasReflection && !hasTransmission)
             || (!hasDirect && !hasScattered))
            return Spectrum(.0f);

        Assert(!!bRec.pltCtx);
        
        Float costheta_t;
        const auto Fr = fresnelDielectricExt(Frame::cosTheta(bRec.wi),costheta_t,m_eta);
        const auto q = m_q->eval(bRec.its).average();
        const auto a = gaussianSurface::alpha(Frame::cosTheta(bRec.wi), q);
        const auto sigma2 = m_sigma2->eval(bRec.its).average();
        
        Vector3 wo;
        
        auto pdf = 1.f;
        auto weight = Spectrum(1.f);
        bool isDirect = hasDirect;
        if (hasDirect && hasScattered) {
            const auto pdfdirect = a.average();
            isDirect = sample.x <= pdfdirect;
            pdf *= isDirect ? pdfdirect : 1.f-pdfdirect;
        }

        bool isReflection = hasReflection;
        if (hasReflection && hasTransmission) {
            if (isDirect) {
                isReflection =  sample.y <= Fr;
                const auto m00 = isReflection ? m_specularReflectance->eval(bRec.its) : 
                                                m_specularTransmittance->eval(bRec.its);
                
                wo = isReflection ? reflect(bRec.wi) : refract(bRec.wi, costheta_t);
                weight *= a * m00;
            }
            else {
                wo = gaussianSurface::sampleScattered(sigma2, *bRec.pltCtx, bRec.wi, *bRec.sampler);
                if (wo.z*bRec.wi.z<.0f)
                    wo.z *= -1.f;
                pdf *= gaussianSurface::scatteredPdf(sigma2, *bRec.pltCtx, bRec.wi, wo);
                
                const auto h = bRec.wi + wo;
                const auto m = normalize(h);
                const auto r = fresnelDielectricExt(dot(bRec.wi,m),m_eta);

                isReflection = sample.y <= r;
                if (!isReflection)
                    wo = scattered_wo(wo);
                
                const auto m00 = isReflection ? m_specularReflectance->eval(bRec.its) : 
                                                m_specularTransmittance->eval(bRec.its);
                weight *= std::fabs(Frame::cosTheta(wo)) * (Spectrum(1.f)-a) * m00;
            }
        }
        else if (hasReflection) {
            const auto m00 = m_specularReflectance->eval(bRec.its);
            if (isDirect) {
                wo = reflect(bRec.wi);
                weight *= a * m00 * Fr;
            }
            else {
                wo = gaussianSurface::sampleScattered(sigma2, *bRec.pltCtx, bRec.wi, *bRec.sampler);
                if (wo.z*bRec.wi.z<.0f)
                    wo.z *= -1.f;
                pdf *= gaussianSurface::scatteredPdf(sigma2, *bRec.pltCtx, bRec.wi, wo);

                const auto h = bRec.wi + wo;
                const auto m = normalize(h);
                const auto r = fresnelDielectricExt(dot(bRec.wi,m),m_eta);
                weight *= std::fabs(Frame::cosTheta(wo)) * (Spectrum(1.f)-a) * m00 * r;
            }
        }
        else {
            const auto m00 = m_specularTransmittance->eval(bRec.its);
            if (isDirect) {
                wo = refract(bRec.wi, costheta_t);
                weight *= a * m00 * (1-Fr);
            }
            else {
                wo = gaussianSurface::sampleScattered(sigma2, *bRec.pltCtx, bRec.wi, *bRec.sampler);
                if (wo.z*bRec.wi.z<.0f)
                    wo.z *= -1.f;
                pdf *= gaussianSurface::scatteredPdf(sigma2, *bRec.pltCtx, bRec.wi, wo);
                
                const auto h = bRec.wi + wo;
                const auto m = normalize(h);
                const auto r = fresnelDielectricExt(dot(bRec.wi,m),m_eta);
                
                wo = scattered_wo(wo);
                weight *= std::fabs(Frame::cosTheta(wo)) * (Spectrum(1.f)-a) * m00 * (1-r);
            }
        }
        
        if (wo.z == 0 || pdf <= RCPOVERFLOW)
            return Spectrum(.0f);

        Assert((isReflection  && wo.z*bRec.wi.z>.0f) ||
               (!isReflection && wo.z*bRec.wi.z<.0f));
        
        bRec.eta = 1.f;
        if (!isReflection)
            bRec.eta = wo.z < 0 ? m_eta : m_invEta;
        bRec.sampledComponent =  isDirect &&  isReflection ? 0 :
                                 isDirect && !isReflection ? 1 :
                                !isDirect &&  isReflection ? 2 : 3;
        bRec.sampledType =  isDirect &&  isReflection ? EDirectReflection :
                            isDirect && !isReflection ? EDirectTransmission :
                           !isDirect &&  isReflection ? EScatteredReflection : EScatteredTransmission;
        bRec.wo = wo;
        return weight / pdf;
    }

    Float getEta() const {
        return m_eta;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "Dielectric[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  eta = " << m_eta << "," << endl
            << "  q = " << indent(m_q->toString()) << "," << endl
            << "  sigma2 = " << indent(m_sigma2->toString()) << "," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "  specularTransmittance = " << indent(m_specularTransmittance->toString()) << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_specularTransmittance;
    ref<Texture> m_specularReflectance;
    ref<Texture> m_q, m_sigma2;
    Float m_eta, m_invEta;
};


MTS_IMPLEMENT_CLASS_S(Dielectric, false, BSDF)
MTS_EXPORT_PLUGIN(Dielectric, "Rough dielectric BSDF");
MTS_NAMESPACE_END
