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

#include <mitsuba/plt/plt.hpp>
#include <mitsuba/plt/birefringence.hpp>

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
    }

    ThinDielectric(Stream *stream, InstanceManager *manager)
            : BSDF(stream, manager) {
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
        if (!m_specularReflectance->isConstant() || !m_specularTransmittance->isConstant())
            extraFlags |= ESpatiallyVarying;
        m_components.clear();
        m_components.push_back(EDirectReflection | EFrontSide | EBackSide | extraFlags);
        m_components.push_back(EDirectTransmission | EFrontSide | EBackSide | extraFlags);

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
        const float ee = eo + birefringence*.2f;
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
        else {
            Assert(false && "Not implemented!");
        }
    }

    inline Vector reflect(const Vector &wi) const {
        return Vector(-wi.x, -wi.y, wi.z);
    }
    inline Vector transmit(const Vector &wi) const {
        return -wi;
    }
    
    Spectrum envelope(const BSDFSamplingRecord &bRec, Float &eta, EMeasure measure) const {
        bool sampleReflection   = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;
        bool sampleTransmission = (bRec.typeMask & EDirectTransmission)
                && (bRec.component == -1 || bRec.component == 1) && measure == EDiscrete;
 
        Assert(!!bRec.pltCtx);

        Float R = fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)), m_eta), T = 1-R;
        if (R < 1)
            R += T*T * R / (1-R*R);

        if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
            if (!sampleReflection || std::abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
                return Spectrum(0.0f);

            return m_specularReflectance->eval(bRec.its) * R;
        } else {
            if (!sampleTransmission || std::abs(dot(transmit(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
                return Spectrum(0.0f);

            return m_specularTransmittance->eval(bRec.its) * (1 - R);
        }
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, Float &eta,
                  RadiancePacket &radPac, EMeasure measure) const { 
        bool sampleReflection   = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;
        bool sampleTransmission = (bRec.typeMask & EDirectTransmission)
                && (bRec.component == -1 || bRec.component == 1) && measure == EDiscrete;
        const auto isReflection = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0;

        if ((!sampleReflection && isReflection) || (!sampleTransmission && !isReflection) 
             || Frame::cosTheta(bRec.wi) == 0)
            return Spectrum(0.0f);
        
        Assert(bRec.mode==EImportance && radPac.isValid());
        Assert(!!bRec.pltCtx);
        
        // Rotate to sp-frame first
        radPac.rotateFrame(bRec.its, Frame::spframe(bRec.wo));

        const auto m00 = isReflection ? m_specularReflectance->eval(bRec.its) : 
                                        m_specularTransmittance->eval(bRec.its);
        
        const auto costheta_i = std::abs(Frame::cosTheta(bRec.wi));
        const auto R = MuellerFresnelRDielectric(costheta_i, m_eta),
                   invOneMinusR = invOneMinusMuellerFresnelRDielectric(costheta_i, m_eta),
                   T = MuellerFresnelTDielectric(costheta_i, m_eta);
        
        Matrix4x4 M;
        if (isReflection) {
            if (std::abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
                return Spectrum(0.0f);
        } else {
            if (std::abs(dot(transmit(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
                return Spectrum(0.0f);
        }
        
        const auto& in = radPac.spectrum();
        Spectrum result = Spectrum(.0f);
        for (std::size_t idx=0; idx<radPac.size(); ++idx) {
            const auto k = Spectrum::ks()[idx];
            if (isReflection) {
                const auto& M = R.m[0][0]<1-Epsilon ? (Matrix4x4)(R + T * R * invOneMinusR * T) : R;
                radPac.L(idx) = m00[idx] * (M * radPac.S(idx));
            }
            else {
                // Transmission effects
                const auto B = m_birefringence->eval(bRec.its).average();
                if (B!=.0f) {
                    auto Lx = radPac.Lx(idx);
                    auto Ly = radPac.Ly(idx);
                    handle_birefringence(Lx, Ly, radPac, *bRec.pltCtx, bRec.its, B, bRec.wi, k, false);

                    radPac.setL(idx, Lx, Ly);
                }
                else {
                    const auto& M = R.m[0][0]<1-Epsilon ? (Matrix4x4)(T * invOneMinusR * T) : T;
                    radPac.L(idx) = m00[idx] * (M * radPac.S(idx));
                }

                // Polarizer
                if (m_polarizer) {
                    const auto& d = radPac.f.toLocal(BSDF::getFrame(bRec.its).toWorld(
                        { std::cos(m_polarizationDir*M_PI/180),std::sin(m_polarizationDir*M_PI/180),0 }));
                    const auto& P = MuellerPolarizer(std::atan2(d.y,d.x));
                    auto L = (Matrix4x4)P * radPac.S(idx);
                    L[0] = std::max(.0f, L[0]);
                    radPac.L(idx) = L;
                }
            }

            if (in[idx]>RCPOVERFLOW)
                result[idx] = radPac.S(idx)[0] / in[idx];
        }

        return result;
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        bool sampleReflection   = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;
        bool sampleTransmission = (bRec.typeMask & EDirectTransmission)
                && (bRec.component == -1 || bRec.component == 1) && measure == EDiscrete;

        Float R = fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)), m_eta), T = 1-R;
        if (R < 1)
            R += T*T * R / (1-R*R);

        if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
            if (!sampleReflection || std::abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
                return 0.0f;

            return sampleTransmission ? R : 1.0f;
        } else {
            if (!sampleTransmission || std::abs(dot(transmit(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
                return 0.0f;

            return sampleReflection ? 1-R : 1.0f;
        }
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        bool sampleReflection   = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0);
        bool sampleTransmission = (bRec.typeMask & EDirectTransmission)
                && (bRec.component == -1 || bRec.component == 1);

        Float R = fresnelDielectricExt(Frame::cosTheta(bRec.wi), m_eta), T = 1-R;
        if (R < 1)
            R += T*T * R / (1-R*R);

        if (sampleTransmission && sampleReflection) {
            if (sample.x <= R) {
                bRec.sampledComponent = 0;
                bRec.sampledType = EDirectReflection;
                bRec.wo = reflect(bRec.wi);
                bRec.eta = 1.0f;

                return m_specularReflectance->eval(bRec.its);
            } else {
                bRec.sampledComponent = 1;
                bRec.sampledType = EDirectTransmission;
                bRec.wo = transmit(bRec.wi);
                bRec.eta = 1.0f;

                return m_specularTransmittance->eval(bRec.its);
            }
        } else if (sampleReflection) {
            bRec.sampledComponent = 0;
            bRec.sampledType = EDirectReflection;
            bRec.wo = reflect(bRec.wi);
            bRec.eta = 1.0f;

            return m_specularReflectance->eval(bRec.its) * R;
        } else if (sampleTransmission) {
            bRec.sampledComponent = 1;
            bRec.sampledType = EDirectTransmission;
            bRec.wo = transmit(bRec.wi);
            bRec.eta = 1.0f;

            return m_specularTransmittance->eval(bRec.its) * (1-R);
        }

        return Spectrum(0.0f);
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
    bool m_polarizer{ false };
    Float m_polarizationDir;
};

MTS_IMPLEMENT_CLASS_S(ThinDielectric, false, BSDF)
MTS_EXPORT_PLUGIN(ThinDielectric, "Thin dielectric BSDF");
MTS_NAMESPACE_END
