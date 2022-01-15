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
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/constants.h>
#include "ior.h"
#include "mitsuba/core/math.h"
#include "mitsuba/render/common.h"

#include <mitsuba/plt/plt.hpp>
#include <mitsuba/plt/birefringence.hpp>
#include <mitsuba/plt/gaussianSurface.hpp>

MTS_NAMESPACE_BEGIN

class ThinDielectric : public BSDF {
public:
    ThinDielectric(const Properties &props) : BSDF(props) {
        /* Specifies the internal index of refraction at the interface */
        Float intIOR = lookupIOR(props, "intIOR", "bk7");

        /* Specifies the external index of refraction at the interface */
        Float extIOR = lookupIOR(props, "extIOR", "air");

        if (intIOR < 0 || extIOR < 0)
            Log(EError, "The interior and exterior indices of "
                "refraction must be positive!");

        m_eta = intIOR / extIOR;
        m_etai = extIOR;
        m_etao = intIOR;
        
        // optic axis and thickness for birefringent matrials
        m_A = props.getVector("opticAxis", Vector3{ 0,0,1 });
        m_A = normalize(m_A);
        m_tau = new ConstantFloatTexture(props.getFloat("thickness", .0f));
        m_birefringence = new ConstantFloatTexture(props.getFloat("birefringence", .0f));
        
        if (props.hasProperty("polarizer")) {
            m_polarizer = true;
            m_polarizationDir = props.getFloat("polarizer");
        }

        m_specularReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("specularReflectance", Spectrum(1.0f)));
        m_specularTransmittance = new ConstantSpectrumTexture(
            props.getSpectrum("specularTransmittance", Spectrum(1.0f)));

        m_q = new ConstantFloatTexture(props.getFloat("q", .0f));
        m_sigma2 = new ConstantFloatTexture(props.getFloat("sigma2", .0f));
    }

    ThinDielectric(Stream *stream, InstanceManager *manager)
            : BSDF(stream, manager) {
        m_q = static_cast<Texture *>(manager->getInstance(stream));
        m_sigma2 = static_cast<Texture *>(manager->getInstance(stream));
        m_eta = stream->readFloat();
        m_etai = stream->readFloat();
        m_etao = stream->readFloat();
        m_A[0] = stream->readFloat();
        m_A[1] = stream->readFloat();
        m_A[2] = stream->readFloat();
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_specularTransmittance = static_cast<Texture *>(manager->getInstance(stream));
        m_birefringence = static_cast<Texture *>(manager->getInstance(stream));
        m_tau = static_cast<Texture *>(manager->getInstance(stream));
        m_polarizer = stream->readBool();
        m_polarizationDir = stream->readFloat();
        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        manager->serialize(stream, m_q.get());
        manager->serialize(stream, m_sigma2.get());
        stream->writeFloat(m_eta);
        stream->writeFloat(m_etai);
        stream->writeFloat(m_etao);
        stream->writeFloat(m_A[0]);
        stream->writeFloat(m_A[1]);
        stream->writeFloat(m_A[2]);
        manager->serialize(stream, m_specularReflectance.get());
        manager->serialize(stream, m_specularTransmittance.get());
        manager->serialize(stream, m_birefringence.get());
        manager->serialize(stream, m_tau.get());
        stream->writeBool(m_polarizer);
        stream->writeFloat(m_polarizationDir);
    }

    void configure() {
        unsigned int extraFlags = 0;
        unsigned int extraFlagsScattered = 0;
        if (!m_q->isConstant() || !m_specularReflectance->isConstant())
            extraFlags |= ESpatiallyVarying;
        if (!m_specularReflectance->isConstant() || !m_specularTransmittance->isConstant())
            extraFlags |= ESpatiallyVarying;
        m_components.clear();
        m_components.push_back(EDirectReflection | EFrontSide | EBackSide | extraFlags);
        m_components.push_back(ENull | EFrontSide | EBackSide | extraFlags);
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
            else if (name == "birefringence")
                m_birefringence = static_cast<Texture *>(child);
            else if (name == "thickness")
                m_tau = static_cast<Texture *>(child);
            else
                BSDF::addChild(name, child);
        } else {
            BSDF::addChild(name, child);
        }
    }
    
    inline void handle_birefringence(Float &Lx, Float &Ly, 
                                     const RadiancePacket &radPac, const PLTContext &pltCtx,
                                     const Intersection &its, Float birefringence, const Vector3 &wi, 
                                     Float k, bool refl) const {
        const auto phii = std::atan2(wi.y,wi.x);
        const auto phiA = std::atan2(m_A.y, m_A.x)-phii;

        const auto I = Vector3{ 0,std::abs(wi.z),std::sqrt(1-sqr(wi.z)) };
        auto A = Vector3{ std::cos(phiA),0,std::sin(phiA) };
        A *= std::sqrt(1-sqr(m_A.z));
        A.y = math::sgn(wi.z) * m_A.z;
        const auto A2 = Vector3{ A.x,-A.y,A.z };
        const auto Z = radPac.f.toLocal(BSDF::getFrame(its).toWorld(Vector3{ std::cos(phii),std::sin(phii),0 }));

        const float ei = m_etai;
        const float eo = m_etao;
        const float ee = eo + birefringence;
        const auto tau = m_tau->eval(its).average() * 1e+6f; // in micron
        
        // Downwards and upwards propagating ordinary and extraordinary vectors in the slab
        Vector3 Io, Io2, Ie, Ie2;
        // Effeective refractive indices of the extraordinary vectors
        float e_eff, e_eff2;
        birefringence::vectors_in_slab(I, ei,eo,ee, A, Io, Io2, Ie, Ie2, e_eff, e_eff2);

        // Fresnel coefficients
        float rss, rsp, tso, tse, rps, rpp, tpo, tpe;
        // float r2oo, r2oe, t2os, t2op, r2eo, r2ee, t2es, t2ep;
        float roo, roe, tos, top, reo, ree, tes, tep;
        birefringence::fresnel_iso_aniso(I.y,I.z, ei,eo,ee, A,  rss, rsp, tso, tse, rps, rpp, tpo, tpe);
        // birefringence::fresnel_aniso_iso(I.y,I.z, ei,eo,ee, A,  r2oo, r2oe, t2os, t2op, r2eo, r2ee, t2es, t2ep);
        birefringence::fresnel_aniso_iso(I.y,I.z, ei,eo,ee, A2, roo, roe, tos, top, reo, ree, tes, tep);
        
        // Offsets
        const float Ioz =  std::abs(tau / Io.y)  * Io.z;
        const float Iez =  std::abs(tau / Ie.y)  * Ie.z;
        // const float Ioz2 = std::abs(tau / Io2.y) * Io2.z;
        // const float Iez2 = std::abs(tau / Ie2.y) * Ie2.z;
        // OPLs
        const float OPLo =  std::abs(tau / Io.y)  * eo;
        const float OPLe =  std::abs(tau / Ie.y)  * e_eff;
        // const float OPLo2 = std::abs(tau / Io2.y) * eo;
        // const float OPLe2 = std::abs(tau / Ie2.y) * e_eff2;

        const auto sqrtLxLy = std::sqrt(Lx*Ly);
        if (!refl) {
            const auto ss = tso*tos + tse*tes;
            const auto sp = tso*top + tse*tep;
            const auto ps = tpo*tos + tpe*tes;
            const auto pp = tpo*top + tpe*tep;
            const auto oez = std::abs(Ioz-Iez);
            const auto mutual_coh = radPac.mutualCoherence(k, oez*Z, pltCtx.sigma_zz * 1e+6f);   // in micron
            const auto t = mutual_coh * std::sin(-k*(ei*oez*I.z+OPLo-OPLe));    // Interference term, modulated by mutual coherence

            const auto nLx = std::max(.0f, sqr(ss)*Lx + sqr(ps)*Ly + 2*ss*ps*t*sqrtLxLy);
            Ly = std::max(.0f, sqr(sp)*Lx + sqr(pp)*Ly + sp*pp*t*sqrtLxLy);
            Lx = nLx;
        }
    }

    inline Vector reflect(const Vector &wi) const {
        return Vector(-wi.x, -wi.y, wi.z);
    }
    inline Vector transmit(const Vector &wi) const {
        return -wi;
    }
    inline Vector scattered_wo(const Vector &wo) const {
        return -reflect(wo);
    }
    
    Spectrum envelope(const BSDFSamplingRecord &bRec, Float &eta, EMeasure measure) const {
        bool sampleReflection   = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;
        bool sampleTransmission = (bRec.typeMask & ENull)
                && (bRec.component == -1 || bRec.component == 1) && measure == EDiscrete;
        bool hasScattered = (bRec.typeMask & EScattered)
                && (bRec.component == -1 || bRec.component == 2 || bRec.component == 3)
                && measure == ESolidAngle;
        const auto isReflection = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) > 0;
        
        if (!isReflection || std::abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
            sampleReflection = false;
        if (isReflection || std::abs(dot(transmit(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
            sampleTransmission = false;
        if (sampleReflection || sampleTransmission)
            hasScattered = false;

        if ((!hasScattered && !sampleReflection && !sampleTransmission) 
             || Frame::cosTheta(bRec.wi) == 0)
            return Spectrum(0.0f);

        const auto q = m_q->eval(bRec.its).average();
        const auto sigma2 = m_sigma2->eval(bRec.its).average();
        const auto a = gaussianSurface::alpha(Frame::cosTheta(bRec.wi), q);
        const auto m00 = isReflection ? m_specularReflectance->eval(bRec.its) : 
                                        m_specularTransmittance->eval(bRec.its);

        Assert(!!bRec.pltCtx);

        if (sampleReflection || sampleTransmission) {
            Float R = fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)), m_eta), T = 1-R;
            if (R < 1)
                R += T*T * R / (1-R*R);

            if (isReflection)
                return a * m00 * R;
            else
                return a * m00 * (1 - R);
        }
        else if (hasScattered) {
            const auto costheta_o = std::fabs(Frame::cosTheta(bRec.wo));
            const auto& wo = isReflection ? bRec.wo : scattered_wo(bRec.wo);

            const auto h = bRec.wi + wo;
            const auto m = normalize(h);
            Float R = fresnelDielectricExt(std::abs(dot(bRec.wi,m)),m_eta), T = 1-R;
            if (R < 1)
                R += T*T * R / (1-R*R);

            return costheta_o * (isReflection ? R : 1-R) *
                    (Spectrum(1.f)-a) * m00 * 
                    gaussianSurface::envelopeScattered(sigma2, *bRec.pltCtx, h);
        }
        
        return Spectrum(.0f);
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, Float &eta,
                  RadiancePacket &rpp, EMeasure measure) const { 
        bool sampleReflection   = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;
        bool sampleTransmission = (bRec.typeMask & ENull)
                && (bRec.component == -1 || bRec.component == 1) && measure == EDiscrete;
        bool hasScattered = (bRec.typeMask & EScattered)
                && (bRec.component == -1 || bRec.component == 2 || bRec.component == 3)
                && measure == ESolidAngle;
        const auto isReflection = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) > 0;

        if (!isReflection || std::abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
            sampleReflection = false;
        if (isReflection || std::abs(dot(transmit(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
            sampleTransmission = false;
        if (sampleReflection || sampleTransmission)
            hasScattered = false;

        if ((!hasScattered && !sampleReflection && !sampleTransmission) 
             || Frame::cosTheta(bRec.wi) == 0)
            return Spectrum(0.0f);
        
        Assert(bRec.mode==EImportance && rpp.isValid());
        Assert(!!bRec.pltCtx);
        
        // Rotate to sp-frame first
        const auto fi = rpp.f;
        Matrix3x3 Q = Matrix3x3(bRec.its.toLocal(fi.s),
                                bRec.its.toLocal(fi.t),
                                bRec.its.toLocal(fi.n)), Qt;
        Q.transpose(Qt);
        
        // Rotate to sp-frame first
        rpp.rotateFrame(bRec.its, Frame::spframe(bRec.wo));
        
        const auto q = m_q->eval(bRec.its).average();
        const auto sigma2 = m_sigma2->eval(bRec.its).average();
        const auto a = gaussianSurface::alpha(Frame::cosTheta(bRec.wi), q).average();
        const auto m00 = isReflection ? m_specularReflectance->eval(bRec.its) : 
                                        m_specularTransmittance->eval(bRec.its);

        const auto B = m_birefringence->eval(bRec.its).average();
        
        Float D = 1, costheta_i;
        if (!hasScattered) {
            costheta_i = std::abs(Frame::cosTheta(bRec.wi));
        }
        else {
            const auto k = Spectrum::ks().average();
            const auto& wo = isReflection ? bRec.wo : -reflect(bRec.wo);
            const auto h = bRec.wi + wo;
            const auto m = normalize(h);
            const auto costheta_o = std::abs(Frame::cosTheta(bRec.wo));

            D = costheta_o * 
                gaussianSurface::diffract(rpp.invTheta(k,bRec.pltCtx->sigma_zz * 1e+6f), 
                                          sigma2, Q,Qt, k*h);
            costheta_i = std::abs(dot(bRec.wi,m));
        }
        
        Matrix4x4 M;
        {
            const auto R = MuellerFresnelRDielectric(costheta_i, m_eta),
                internalRs = invOneMinusMuellerFresnelRDielectric(costheta_i, m_eta),
                T = MuellerFresnelTDielectric(costheta_i, m_eta);
            M = isReflection ? 
                    (internalRs.m[0][0]>0 ? (Matrix4x4)(R + T * R * internalRs * T) : (Matrix4x4)R) :
                    (internalRs.m[0][0]>0 ? (Matrix4x4)(T * internalRs * T)         : (Matrix4x4)(T*T));
        }
        
        const auto in = rpp.spectrum();
        Spectrum result = Spectrum(.0f);
        for (std::size_t idx=0; idx<rpp.size(); ++idx) {
            const auto k = Spectrum::ks()[idx];
            if (B!=.0f && !isReflection) {
                auto Lx = rpp.Lx(idx);
                auto Ly = rpp.Ly(idx);
                handle_birefringence(Lx, Ly, rpp, *bRec.pltCtx, bRec.its, B, bRec.wi, k, isReflection);

                rpp.setL(idx, Lx, Ly);
            }
            else {
                rpp.L(idx) = (M * rpp.S(idx));
            }
            
            rpp.L(idx) = D * m00[idx] * (hasScattered ? 1-a : a) * rpp.S(idx);

            if (!isReflection) {
                // Polarizer
                if (m_polarizer) {
                    const auto& d = rpp.f.toLocal(BSDF::getFrame(bRec.its).toWorld(
                        { std::cos(m_polarizationDir*M_PI/180),std::sin(m_polarizationDir*M_PI/180),0 }));
                    const auto& P = MuellerPolarizer(std::atan2(d.y,d.x));
                    auto L = (Matrix4x4)P * rpp.S(idx);
                    L[0] = std::max(.0f, L[0]);
                    rpp.L(idx) = L;
                }
            }
            
            if (rpp.L(idx)[0]<=0)
                rpp.L(idx) = {0,0,0,0};

            if (in[idx]>RCPOVERFLOW)
                result[idx] = rpp.S(idx)[0] / in[idx];
        }

        return result;
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        const auto hasDirect = (bRec.typeMask & EDelta)
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
            Float R = fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)),m_eta), T = 1-R;
            if (R < 1)
                R += T*T * R / (1-R*R);

            Float F = 1.f;
            if (hasReflection && hasTransmission)
                F = isReflection ? R : 1-R;

            if ((isReflection  && std::abs(dot(reflect(bRec.wi), bRec.wo)-1) <= DeltaEpsilon) ||
                (!isReflection && std::abs(dot(transmit(bRec.wi), bRec.wo)-1) <= DeltaEpsilon))
                return F * probDirect;
        }
        if (hasScattered && measure == ESolidAngle && probDirect<1) {
            const auto& wo = isReflection ? bRec.wo : scattered_wo(bRec.wo);
            if (wo.z == 0)
                return 0;
            const auto sigma2 = m_sigma2->eval(bRec.its).average();
            const auto h = bRec.wi + wo;
            const auto m = normalize(h);
            
            auto R = fresnelDielectricExt(std::abs(dot(bRec.wi,m)),m_eta), T = 1-R;
            if (R < 1)
                R += T*T * R / (1-R*R);

            Float F = 1.f;
            if (hasReflection && hasTransmission)
                F = isReflection ? R : 1.f-R;
            
            return gaussianSurface::scatteredPdf(sigma2, *bRec.pltCtx, bRec.wi, wo) * 
                F * (1-probDirect);
        }

        return 0;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        const auto hasDirect = (bRec.typeMask & EDelta)
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
        
        Float R = fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)),m_eta), T=1-R;
        if (R < 1)
            R += T*T * R / (1-R*R);
        
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
                isReflection =  sample.y <= R;
                const auto m00 = isReflection ? m_specularReflectance->eval(bRec.its) : 
                                                m_specularTransmittance->eval(bRec.its);
                
                wo = isReflection ? reflect(bRec.wi) : transmit(bRec.wi);
                weight *= a * m00;
            }
            else {
                wo = gaussianSurface::sampleScattered(sigma2, *bRec.pltCtx, bRec.wi, *bRec.sampler);
                if (wo.z*bRec.wi.z<.0f)
                    wo.z *= -1.f;
                pdf *= gaussianSurface::scatteredPdf(sigma2, *bRec.pltCtx, bRec.wi, wo);
                
                const auto h = bRec.wi + wo;
                const auto m = normalize(h);
                Float r = fresnelDielectricExt(std::abs(dot(bRec.wi,m)),m_eta), t=1-r;
                if (r < 1)
                    r += t*t * r / (1-r*r);

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
                weight *= a * m00 * R;
            }
            else {
                wo = gaussianSurface::sampleScattered(sigma2, *bRec.pltCtx, bRec.wi, *bRec.sampler);
                if (wo.z*bRec.wi.z<.0f)
                    wo.z *= -1.f;
                pdf *= gaussianSurface::scatteredPdf(sigma2, *bRec.pltCtx, bRec.wi, wo);

                const auto h = bRec.wi + wo;
                const auto m = normalize(h);
                const auto r = fresnelDielectricExt(std::abs(dot(bRec.wi,m)),m_eta);
                weight *= std::fabs(Frame::cosTheta(wo)) * (Spectrum(1.f)-a) * m00 * r;
            }
        }
        else {
            const auto m00 = m_specularTransmittance->eval(bRec.its);
            if (isDirect) {
                wo = transmit(bRec.wi);
                weight *= a * m00 * (1-R);
            }
            else {
                wo = gaussianSurface::sampleScattered(sigma2, *bRec.pltCtx, bRec.wi, *bRec.sampler);
                if (wo.z*bRec.wi.z<.0f)
                    wo.z *= -1.f;
                pdf *= gaussianSurface::scatteredPdf(sigma2, *bRec.pltCtx, bRec.wi, wo);
                
                const auto h = bRec.wi + wo;
                const auto m = normalize(h);
                Float r = fresnelDielectricExt(std::abs(dot(bRec.wi,m)),m_eta), t=1-r;
                if (r < 1)
                    r += t*t * r / (1-r*r);
                
                wo = scattered_wo(wo);
                weight *= std::fabs(Frame::cosTheta(wo)) * (Spectrum(1.f)-a) * m00 * (1-r);
            }
        }
        
        if (wo.z == 0 || pdf <= RCPOVERFLOW)
            return Spectrum(.0f);

        Assert((isReflection  && wo.z*bRec.wi.z>.0f) ||
               (!isReflection && wo.z*bRec.wi.z<.0f));
        
        bRec.eta = 1.f;
        bRec.sampledComponent =  isDirect &&  isReflection ? 0 :
                                 isDirect && !isReflection ? 1 :
                                !isDirect &&  isReflection ? 2 : 3;
        bRec.sampledType =  isDirect &&  isReflection ? EDirectReflection :
                            isDirect && !isReflection ? ENull :
                           !isDirect &&  isReflection ? EScatteredReflection : EScatteredTransmission;
        bRec.wo = wo;
        return weight / pdf;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "ThinDielectric[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  eta = " << m_eta << "," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "  specularTransmittance = " << indent(m_specularTransmittance->toString()) << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    Float m_eta, m_etai, m_etao;
    Vector3 m_A;
    ref<Texture> m_specularTransmittance;
    ref<Texture> m_specularReflectance;
    ref<Texture> m_birefringence, m_tau;
    ref<Texture> m_q, m_sigma2;
    bool m_polarizer{ false };
    Float m_polarizationDir;
};

MTS_IMPLEMENT_CLASS_S(ThinDielectric, false, BSDF)
MTS_EXPORT_PLUGIN(ThinDielectric, "Thin dielectric BSDF");
MTS_NAMESPACE_END
