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
#include "mitsuba/core/constants.h"
#include "mitsuba/core/util.h"
#include "mitsuba/plt/mueller.hpp"
#include "mitsuba/render/common.h"
#include <mitsuba/plt/plt.hpp>
#include "ior.h"

#include <mitsuba/plt/gaussianFractalSurface.hpp>

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

        const auto q = props.getFloat("q", .0f);
        gfs.gamma = props.getFloat("gamma", 3.f);
        if (props.hasProperty("T"))
            gfs.T = props.getFloat("T");
        else 
            gfs.T = props.getFloat("sigma2",.0f) / (gfs.gamma+1);
        gfs.sigma_h2 = q;
    }

    Dielectric(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_specularTransmittance = static_cast<Texture *>(manager->getInstance(stream));
        m_eta = stream->readFloat();
        gfs.gamma = stream->readFloat();
        gfs.T = stream->readFloat();
        gfs.sigma_h2 = stream->readFloat();

        m_invEta = 1.f / m_eta;

        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        manager->serialize(stream, m_specularReflectance.get());
        manager->serialize(stream, m_specularTransmittance.get());
        stream->writeFloat(m_eta);
        stream->writeFloat(gfs.gamma);
        stream->writeFloat(gfs.T);
        stream->writeFloat(gfs.sigma_h2);
    }

    void configure() {
        unsigned int extraFlags = 0;
        unsigned int extraFlagsScattered = 0;
        if (!m_specularReflectance->isConstant())
            extraFlags |= ESpatiallyVarying;

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
            if (name == "specularReflectance")
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
    // Cheap lame approximation
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
        const auto costheta_i = Frame::cosTheta(bRec.wi);
        const auto costheta_o = Frame::cosTheta(bRec.wo);
        const auto isReflection = costheta_i * costheta_o > 0;

        if ((!hasDirect && !hasScattered) || costheta_i == 0 ||
            (isReflection && bRec.component != -1 && bRec.component != 0 && bRec.component != 2))
            return Spectrum(0.0f);
        
        Assert(!!bRec.pltCtx);
        
        const auto m00 = isReflection ? m_specularReflectance->eval(bRec.its) : 
                                        m_specularTransmittance->eval(bRec.its);
        const auto a = gfs.alpha(costheta_i, costheta_o);

        if (!hasDirect && a.average()==1)
            return Spectrum(.0f);

        const auto& wo = isReflection ? bRec.wo : scattered_wo(bRec.wo);
        Assert(wo.z*bRec.wi.z>=.0f);
        
        if (wo.z == 0)
            return Spectrum(.0f);
        
        eta = isReflection ? 1.0f : bRec.wo.z<.0f ? m_eta : m_invEta;
        if (hasDirect) {
            Float costheta_t;
            const auto Fr = fresnelDielectricExt(costheta_i,costheta_t,m_eta);

            if (isReflection  && std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
                return Fr * a * m00;
            if (!isReflection && std::abs(dot(refract(bRec.wi, costheta_t), bRec.wo)-1) < DeltaEpsilon)
                return (1-Fr) * a * m00;
        }
        if (hasScattered) {
            const auto h = bRec.wi + wo;
            const auto m = normalize(h);
            const auto r = fresnelDielectricExt(glm::sign(bRec.wi.z) * dot(bRec.wi,m),m_eta);

            return std::fabs(costheta_o) * (isReflection ? r : 1.f-r) *
                    (Spectrum(1.f)-a) * m00 * 
                    gfs.envelopeScattered(*bRec.pltCtx, h);
        }
        return Spectrum(.0f);
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, Float &eta,
                  RadiancePacket &rpp, EMeasure measure) const { 
        const auto hasDirect = (bRec.typeMask & EDirect)
                && (bRec.component == -1 || bRec.component == 0 || bRec.component == 1)
                && measure == EDiscrete;
        const auto hasScattered = (bRec.typeMask & EScattered)
                && (bRec.component == -1 || bRec.component == 2 || bRec.component == 3)
                && measure == ESolidAngle;
        const auto costheta_i = Frame::cosTheta(bRec.wi);
        const auto costheta_o = Frame::cosTheta(bRec.wo);
        const auto isReflection = costheta_i * costheta_o > 0;

        if ((!hasDirect && !hasScattered) || costheta_i == 0 ||
            (isReflection && bRec.component != -1 && bRec.component != 0 && bRec.component != 2))
            return Spectrum(0.0f);
        
        Assert(bRec.mode==EImportance && rpp.isValid());
        Assert(!!bRec.pltCtx);
        
        const auto m00 = isReflection ? m_specularReflectance->eval(bRec.its) : 
                                        m_specularTransmittance->eval(bRec.its);
        const auto a = gfs.alpha(costheta_i, costheta_o);
        const auto& wo = isReflection ? bRec.wo : scattered_wo(bRec.wo);
        const auto h = bRec.wi + wo;
        const auto phii = std::fabs(costheta_i)<1.f ? std::atan2(bRec.wi.y,bRec.wi.x) : .0f;
        const auto phio = std::fabs(costheta_o)<1.f ? std::atan2(bRec.wo.y,bRec.wo.x) : .0f;
        
        if (wo.z == 0)
            return Spectrum(.0f);
        Assert(wo.z*bRec.wi.z>=.0f);
        
        if (!hasDirect && a.average() >= 1-Epsilon)
            return Spectrum(.0f);
        if (hasDirect) {
            Float costheta_t;
            fresnelDielectricExt(costheta_i,costheta_t,m_eta);

            if ((isReflection  && std::abs(dot(reflect(bRec.wi), bRec.wo)-1) >= DeltaEpsilon) ||
                (!isReflection && std::abs(dot(refract(bRec.wi, costheta_t), bRec.wo)-1) >= DeltaEpsilon))
                return Spectrum(.0f);
        }
        
        const auto fi = rpp.f;
        Matrix3x3 Q = Matrix3x3(bRec.its.toLocal(fi.s),
                                bRec.its.toLocal(fi.t),
                                bRec.its.toLocal(fi.n)), Qt;
        Q.transpose(Qt);
        
        // Rotate to sp-frame first
        rpp.rotateFrame(bRec.its, Frame::spframe(bRec.wo));

        const auto k = Spectrum::ks().average();
        const auto D = !hasDirect ? 
                       gfs.diffract(rpp.invTheta(k,bRec.pltCtx->sigma_zz * 1e+6f), 
                                                 Q,Qt, k*h) : .0f;
        const auto M = !hasDirect ? 
                    MuellerFresnelspm1(costheta_i, costheta_o, phii, phio, m_eta, isReflection) :
                    MuellerFresnelDielectric(costheta_i, m_eta, isReflection);
        
        const auto in = rpp.spectrum();
        Spectrum result = Spectrum(.0f);
        for (std::size_t idx=0; idx<rpp.size(); ++idx) {
            auto L = m00[idx]*((Matrix4x4)M * rpp.S(idx));
            L *= hasDirect ?
                a[idx] :
                (std::fabs(costheta_o) * (1-a[idx]) * D);

            if (L.x<=.0f) {
                L = {};
                result[idx] = .0f;
            }
            else {
                if (in[idx]>RCPOVERFLOW)
                    result[idx] = L[0] / in[idx];
                rpp.L(idx) = L;
            }
        }

        eta = isReflection ? 1.0f : bRec.wo.z<.0f ? m_eta : m_invEta;
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
        const auto costheta_i = Frame::cosTheta(bRec.wi);
        const auto costheta_o = Frame::cosTheta(bRec.wo);
        const auto isReflection = costheta_i * costheta_o > 0;

        if (costheta_i == 0 || (!hasReflection && !hasTransmission) || (!hasDirect && !hasScattered) 
                || (isReflection && !hasReflection)|| (!isReflection && !hasTransmission))
            return .0f;

        Assert(!!bRec.pltCtx);

        Float probDirect = hasDirect ? 1.f : .0f;
        if (hasDirect && hasScattered) {
            const auto a = gfs.alpha(costheta_i, costheta_o);
            probDirect = a.average();
        }

        if (hasDirect && measure == EDiscrete && probDirect>0) {
            Float costheta_t;
            const auto Fr = fresnelDielectricExt(costheta_i,costheta_t,m_eta);

            Float F = 1.f;
            if (hasReflection && hasTransmission)
                F = isReflection ? Fr : 1.f-Fr;

            if ((isReflection  && std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon) ||
                (!isReflection && std::abs(dot(refract(bRec.wi, costheta_t), bRec.wo)-1) < DeltaEpsilon))
                return probDirect*F;
        } else if (hasScattered && measure == ESolidAngle && probDirect<1) {
            auto wo = isReflection ? bRec.wo : scattered_wo(bRec.wo);
            if (wo.z*bRec.wi.z<.0f)
                wo.z *= -1.f;
            if (wo.z == 0)
                return .0f;
            const auto h = bRec.wi + wo;
            const auto m = normalize(h);
            
            const auto r = fresnelDielectricExt(glm::sign(bRec.wi.z) * dot(bRec.wi,m),m_eta);
            Float F = 1.f;
            if (hasReflection && hasTransmission)
                F = isReflection ? r : 1.f-r;
            
            return gfs.scatteredPdf(*bRec.pltCtx, bRec.wi, wo) * (1-probDirect) * F;
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

        const auto costheta_i = Frame::cosTheta(bRec.wi);
        const auto costheta_o = Frame::cosTheta(bRec.wo);
        if (costheta_i == 0 || (!hasReflection && !hasTransmission)
             || (!hasDirect && !hasScattered))
            return Spectrum(.0f);

        Assert(!!bRec.pltCtx);
        
        Float costheta_t;
        const auto Fr = fresnelDielectricExt(costheta_i,costheta_t,m_eta);
        const auto a = gfs.alpha(costheta_i, costheta_o);
        
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
                const auto m00 = isReflection ? m_specularReflectance->eval(bRec.its) : m_specularTransmittance->eval(bRec.its);
                
                wo = isReflection ? reflect(bRec.wi) : refract(bRec.wi, costheta_t);
                weight *= a * m00;
            }
            else {
                wo = gfs.sampleScattered(*bRec.pltCtx, bRec.wi, *bRec.sampler);
                if (wo.z*bRec.wi.z<.0f)
                    wo.z *= -1.f;
                pdf *= gfs.scatteredPdf(*bRec.pltCtx, bRec.wi, wo);
                
                const auto h = bRec.wi + wo;
                const auto m = normalize(h);
                const auto r = fresnelDielectricExt(glm::sign(bRec.wi.z) * dot(bRec.wi,m),m_eta);

                isReflection = sample.y <= r;
                if (!isReflection)
                    wo = scattered_wo(wo);
                
                const auto m00 = isReflection ? m_specularReflectance->eval(bRec.its) : 
                                                m_specularTransmittance->eval(bRec.its);
                weight *= std::fabs(costheta_o) * (Spectrum(1.f)-a) * m00;
            }
        }
        else if (hasReflection) {
            const auto m00 = m_specularReflectance->eval(bRec.its);
            if (isDirect) {
                wo = reflect(bRec.wi);
                weight *= a * m00 * Fr;
            }
            else {
                wo = gfs.sampleScattered(*bRec.pltCtx, bRec.wi, *bRec.sampler);
                if (wo.z*bRec.wi.z<.0f)
                    wo.z *= -1.f;
                pdf *= gfs.scatteredPdf(*bRec.pltCtx, bRec.wi, wo);

                const auto h = bRec.wi + wo;
                const auto m = normalize(h);
                const auto r = fresnelDielectricExt(glm::sign(bRec.wi.z) * dot(bRec.wi,m),m_eta);
                weight *= std::fabs(costheta_o) * (Spectrum(1.f)-a) * m00 * r;
            }
        }
        else {
            const auto m00 = m_specularTransmittance->eval(bRec.its);
            if (isDirect) {
                wo = refract(bRec.wi, costheta_t);
                weight *= a * m00 * (1-Fr);
            }
            else {
                wo = gfs.sampleScattered(*bRec.pltCtx, bRec.wi, *bRec.sampler);
                if (wo.z*bRec.wi.z<.0f)
                    wo.z *= -1.f;
                pdf *= gfs.scatteredPdf(*bRec.pltCtx, bRec.wi, wo);
                
                const auto h = bRec.wi + wo;
                const auto m = normalize(h);
                const auto r = fresnelDielectricExt(glm::sign(bRec.wi.z) * dot(bRec.wi,m),m_eta);
                
                wo = scattered_wo(wo);
                weight *= std::fabs(costheta_o) * (Spectrum(1.f)-a) * m00 * (1-r);
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

    Spectrum getSpecularReflectance(const Intersection &its) const override {
        return m_specularReflectance->eval(its);
    }
    virtual void getRefractiveIndex(const Intersection &its, Spectrum &n, Spectrum &k) const override {
        n = Spectrum(m_eta);
        k = Spectrum(.0f);
    }
    Float getEta() const override {
        return m_eta;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "Dielectric[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  eta = " << m_eta << "," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "  specularTransmittance = " << indent(m_specularTransmittance->toString()) << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_specularTransmittance;
    ref<Texture> m_specularReflectance;
    Float m_eta, m_invEta;
    gaussian_fractal_surface gfs;
};


MTS_IMPLEMENT_CLASS_S(Dielectric, false, BSDF)
MTS_EXPORT_PLUGIN(Dielectric, "Rough dielectric BSDF");
MTS_NAMESPACE_END
