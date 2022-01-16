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

#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include "ior.h"

MTS_NAMESPACE_BEGIN

// A simple, perfect, smooth dielectric coat. 
// Assumed to be too thick for interference effects.
class ClearCoat : public BSDF {
public:
    ClearCoat(const Properties &props)
            : BSDF(props) {
        /* Specifies the internal index of refraction at the interface */
        Float intIOR = lookupIOR(props, "intIOR", "bk7");

        /* Specifies the external index of refraction at the interface */
        Float extIOR = lookupIOR(props, "extIOR", "air");

        if (intIOR < 0 || extIOR < 0 || intIOR == extIOR)
            Log(EError, "The interior and exterior indices of "
                "refraction must be positive and differ!");

        m_eta = intIOR / extIOR;
        m_invEta = 1 / m_eta;

        /* Specifies the layer's thickness using the inverse units of sigmaA */
        m_thickness = props.getFloat("thickness", 1);

        /* Specifies the absorption within the layer */
        m_sigmaA = new ConstantSpectrumTexture(
            props.getSpectrum("sigmaA", Spectrum(0.0f)));

        /* Specifies a multiplier for the specular reflectance component */
        m_specularReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("specularReflectance", Spectrum(1.0f)));
    }

    ClearCoat(Stream *stream, InstanceManager *manager)
            : BSDF(stream, manager) {
        m_eta = stream->readFloat();
        m_thickness = stream->readFloat();
        m_nested = static_cast<BSDF *>(manager->getInstance(stream));
        m_sigmaA = static_cast<Texture *>(manager->getInstance(stream));
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_invEta = 1 / m_eta;
        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        stream->writeFloat(m_eta);
        stream->writeFloat(m_thickness);
        manager->serialize(stream, m_nested.get());
        manager->serialize(stream, m_sigmaA.get());
        manager->serialize(stream, m_specularReflectance.get());
    }

    void configure() {
        if (!m_nested)
            Log(EError, "A child BSDF instance is required");

        unsigned int extraFlags = 0;
        if (!m_sigmaA->isConstant())
            extraFlags |= ESpatiallyVarying;

        m_components.clear();
        for (int i=0; i<m_nested->getComponentCount(); ++i)
            m_components.push_back(m_nested->getType(i) | extraFlags);

        m_components.push_back(EDirectReflection | EFrontSide | EBackSide
            | (m_specularReflectance->isConstant() ? 0 : ESpatiallyVarying));

        /* Compute weights that further steer samples towards
           the specular or nested components */
        Float avgAbsorption = (m_sigmaA->getAverage()
             *(-2*m_thickness)).exp().average();

        m_specularSamplingWeight = 1.0f / (avgAbsorption + 1.0f);

        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);

        BSDF::configure();
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(BSDF))) {
            if (m_nested != NULL)
                Log(EError, "Only a single nested BRDF can be added!");
            m_nested = static_cast<BSDF *>(child);
        } else if (child->getClass()->derivesFrom(MTS_CLASS(Texture)) && name == "sigmaA") {
            m_sigmaA = static_cast<Texture *>(child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    /// Reflection in local coordinates
    inline Vector reflect(const Vector &wi) const {
        return Vector(-wi.x, -wi.y, wi.z);
    }

    /// Refract into the material, preserve sign of direction
    inline Vector refractIn(const Vector &wi, Float &R) const {
        Float cosThetaT;
        R = fresnelDielectricExt(std::abs(Frame::cosTheta(wi)), cosThetaT, m_eta);
        return Vector(m_invEta*wi.x, m_invEta*wi.y, -math::signum(Frame::cosTheta(wi)) * cosThetaT);
    }
    inline Vector refractIn(const Vector &wi) const {
        Float unused;
        return refractIn(wi,unused);
    }

    /// Refract out of the material, preserve sign of direction
    inline Vector refractOut(const Vector &wi, Float &R) const {
        Float cosThetaT;
        R = fresnelDielectricExt(std::abs(Frame::cosTheta(wi)), cosThetaT, m_invEta);
        return Vector(m_eta*wi.x, m_eta*wi.y, -math::signum(Frame::cosTheta(wi)) * cosThetaT);
    }

    Spectrum envelope(const BSDFSamplingRecord &bRec, Float &eta, EMeasure measure) const {
        bool sampleSpecular = (bRec.typeMask & EDirectReflection)
            && (bRec.component == -1 || bRec.component == (int) m_components.size()-1);
        bool sampleNested = (bRec.typeMask & m_nested->getType() & BSDF::EAll)
            && (bRec.component == -1 || bRec.component < (int) m_components.size()-1);

        if (measure == EDiscrete && sampleSpecular &&
                std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon) {
            return m_specularReflectance->eval(bRec.its) *
                fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)), m_eta);
        } else if (sampleNested) {
            Float R12, R21;
            BSDFSamplingRecord bRecInt(bRec);
            bRecInt.wi = refractIn(bRec.wi, R12);
            bRecInt.wo = refractIn(bRec.wo, R21);

            if (R12 == 1 || R21 == 1) /* Total internal reflection */
                return Spectrum(0.0f);

            Spectrum result = m_nested->envelope(bRecInt, eta, measure) * (1-R12) * (1-R21);

            Spectrum sigmaA = m_sigmaA->eval(bRec.its) * m_thickness;
            if (!sigmaA.isZero())
                result *= (-sigmaA *
                    (1/std::abs(Frame::cosTheta(bRecInt.wi)) +
                     1/std::abs(Frame::cosTheta(bRecInt.wo)))).exp();

            /* Solid angle compression & irradiance conversion factors */
            if (measure == ESolidAngle)
                result *= //sqr(m_invEta) *
                    Frame::cosTheta(bRec.wo) / Frame::cosTheta(bRecInt.wo);

            return result;
        }

        return Spectrum(.0f);
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, Float &eta,
                  RadiancePacket &rpp, EMeasure measure) const { 
        bool sampleSpecular = (bRec.typeMask & EDirectReflection)
            && (bRec.component == -1 || bRec.component == (int) m_components.size()-1);
        bool sampleNested = (bRec.typeMask & m_nested->getType() & BSDF::EAll)
            && (bRec.component == -1 || bRec.component < (int) m_components.size()-1);

        if (measure == EDiscrete && sampleSpecular &&
                std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon) { 
            // Rotate to exitant sp-frame
            rpp.rotateFrame(bRec.its, Frame::spframe(bRec.wo));
            
            const auto m00 = m_specularReflectance->eval(bRec.its);
            const auto M = MuellerFresnelDielectric(Frame::cosTheta(bRec.wi), m_eta, true);

            const auto in = rpp.spectrum();
            Spectrum result = Spectrum(.0f);
            for (std::size_t idx=0; idx<rpp.size(); ++idx) {
                auto L = m00[idx] * ((Matrix4x4)M * rpp.S(idx));
                if (in[idx]>RCPOVERFLOW)
                    result[idx] = L[0] / in[idx];
                rpp.L(idx) = L;
            }

            return result;
        } else if (sampleNested) {
            BSDFSamplingRecord bRecInt(bRec);
            bRecInt.wi = refractIn(bRec.wi);
            bRecInt.wo = refractIn(bRec.wo);
            
            const auto in = rpp.spectrum();
            
            {
                // Rotate to the refracted frame
                rpp.rotateFrame(bRec.its, Frame::spframe(-bRecInt.wi));
                // Refract into the coat
                const auto Tin = MuellerFresnelDielectric(Frame::cosTheta(bRec.wi), m_eta, false);
                for (std::size_t idx=0; idx<rpp.size(); ++idx) 
                    rpp.L(idx) = (Matrix4x4)Tin * rpp.S(idx);
            }

            // Eval nested BSDF
            m_nested->eval(bRecInt, rpp, measure);
            
            {
                // Rotate to the exitant frame
                rpp.rotateFrame(bRec.its, Frame::spframe(bRec.wo));
                // Refract out of the coat
                const auto Tout = MuellerFresnelDielectric(Frame::cosTheta(-bRecInt.wo), m_eta, false);
                for (std::size_t idx=0; idx<rpp.size(); ++idx)
                    rpp.L(idx) = (Matrix4x4)Tout * rpp.S(idx);
            }
            
            auto terms = Spectrum(1.f);
            // Absorption
            Spectrum sigmaA = m_sigmaA->eval(bRec.its) * m_thickness;
            if (!sigmaA.isZero())
                terms *= (-sigmaA *
                    (1/std::abs(Frame::cosTheta(bRecInt.wi)) +
                     1/std::abs(Frame::cosTheta(bRecInt.wo)))).exp();
            /* Solid angle compression & irradiance conversion factors */
            if (measure == ESolidAngle)
                terms *= //sqr(m_invEta) *
                    Frame::cosTheta(bRec.wo) / Frame::cosTheta(bRecInt.wo);
            
            Spectrum result = Spectrum(.0f);
            for (std::size_t idx=0; idx<rpp.size(); ++idx) {
                auto L = terms[idx] * rpp.S(idx);
                if (in[idx]>RCPOVERFLOW)
                    result[idx] = L[0] / in[idx];
                rpp.L(idx) = L;
            }

            return result;
        }

        return Spectrum(.0f);
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        bool sampleSpecular = (bRec.typeMask & EDirectReflection)
            && (bRec.component == -1 || bRec.component == (int) m_components.size()-1);
        bool sampleNested = (bRec.typeMask & m_nested->getType() & BSDF::EAll)
            && (bRec.component == -1 || bRec.component < (int) m_components.size()-1);

        Float R12;
        Vector wiPrime = refractIn(bRec.wi, R12);

        /* Reallocate samples */
        Float probSpecular = (R12*m_specularSamplingWeight) /
            (R12*m_specularSamplingWeight +
            (1-R12) * (1-m_specularSamplingWeight));

        if (measure == EDiscrete && sampleSpecular &&
                std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon) {
            return sampleNested ? probSpecular : 1.0f;
        } else if (sampleNested) {
            Float R21;
            BSDFSamplingRecord bRecInt(bRec);
            bRecInt.wi = wiPrime;
            bRecInt.wo = refractIn(bRec.wo, R21);

            if (R12 == 1 || R21 == 1) /* Total internal reflection */
                return 0.0f;

            Float pdf = m_nested->pdf(bRecInt, measure);

            if (measure == ESolidAngle)
                pdf *= //sqr(m_invEta) * 
                    Frame::cosTheta(bRec.wo) / Frame::cosTheta(bRecInt.wo);

            return sampleSpecular ? (pdf * (1 - probSpecular)) : pdf;
        } else {
            return 0.0f;
        }
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &_sample) const {
        bool sampleSpecular = (bRec.typeMask & EDirectReflection)
            && (bRec.component == -1 || bRec.component == (int) m_components.size()-1);
        bool sampleNested = (bRec.typeMask & m_nested->getType() & BSDF::EAll)
            && (bRec.component == -1 || bRec.component < (int) m_components.size()-1);

        if ((!sampleSpecular && !sampleNested))
            return Spectrum(0.0f);

        Float R12;
        const auto& wiPrime = refractIn(bRec.wi, R12);

        /* Reallocate samples */
        const auto probSpecular = (R12*m_specularSamplingWeight) /
            (R12*m_specularSamplingWeight +
            (1-R12) * (1-m_specularSamplingWeight));

        bool choseSpecular = sampleSpecular;
        auto sample = _sample;
        if (sampleSpecular && sampleNested) {
            if (sample.x < probSpecular) {
                sample.x /= probSpecular;
            } else {
                sample.x = (sample.x - probSpecular) / (1 - probSpecular);
                choseSpecular = false;
            }
        }

        bRec.eta = 1.0f;
        if (choseSpecular) {
            bRec.sampledComponent = (int) m_components.size() - 1;
            bRec.sampledType = EDirectReflection;
            bRec.wo = reflect(bRec.wi);
            const auto pdf = sampleNested ? probSpecular : 1.0f;
            return m_specularReflectance->eval(bRec.its) * (R12/pdf);
        } else {
            if (R12 == 1.0f) /* Total internal reflection */
                return Spectrum(0.0f);

            Vector wiBackup = bRec.wi;
            bRec.wi = wiPrime;
            Spectrum result = m_nested->sample(bRec, sample);
            bRec.wi = wiBackup;
            if (result.isZero())
                return Spectrum(0.0f);

            Vector woPrime = bRec.wo;

            Spectrum sigmaA = m_sigmaA->eval(bRec.its) * m_thickness;
            if (!sigmaA.isZero())
                result *= (-sigmaA *
                    (1/std::abs(Frame::cosTheta(wiPrime)) +
                     1/std::abs(Frame::cosTheta(woPrime)))).exp();

            Float R21;
            bRec.wo = refractOut(woPrime, R21);
            if (R21 == 1.0f) /* Total internal reflection */
                return Spectrum(0.0f);

            if (sampleSpecular)
                result /= 1.0f - probSpecular;

            result *= (1 - R12) * (1 - R21);

            return result;
        }
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "ClearCoat[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  eta = " << m_eta << "," << endl
            << "  specularSamplingWeight = " << m_specularSamplingWeight << "," << endl
            << "  sigmaA = " << indent(m_sigmaA->toString()) << "," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "  thickness = " << m_thickness << "," << endl
            << "  nested = " << indent(m_nested.toString()) << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    Float m_specularSamplingWeight;
    Float m_eta, m_invEta;
    ref<Texture> m_sigmaA;
    ref<Texture> m_specularReflectance;
    ref<BSDF> m_nested;
    Float m_thickness;
};


MTS_IMPLEMENT_CLASS_S(ClearCoat, false, BSDF)
MTS_EXPORT_PLUGIN(ClearCoat, "Smooth dielectric coating");
MTS_NAMESPACE_END
