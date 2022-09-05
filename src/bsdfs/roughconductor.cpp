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
#include <mitsuba/plt/mueller.hpp>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/core/constants.h>

#include <mitsuba/plt/gaussianFractalSurface.hpp>

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

        const auto q = props.getFloat("q", .0f);
        gfs.gamma = props.getFloat("gamma", 3.f);
        if (props.hasProperty("T"))
            gfs.T = props.getFloat("T");
        else 
            gfs.T = props.getFloat("sigma2",.0f) / (gfs.gamma+1);
        gfs.sigma_h2 = q;
    }

    Conductor(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_eta = Spectrum(stream);
        m_k = Spectrum(stream);
        gfs.gamma = stream->readFloat();
        gfs.T = stream->readFloat();
        gfs.sigma_h2 = stream->readFloat();

        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        m_eta.serialize(stream);
        m_k.serialize(stream);
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
        m_components.push_back(EDirectReflection | EFrontSide | extraFlags);
        m_components.push_back(EScatteredReflection | EFrontSide | EUsesSampler | extraFlags | extraFlagsScattered);

        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);

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
    
    Spectrum envelope(const BSDFSamplingRecord &bRec, Float &eta, EMeasure measure) const {
        const auto hasDirect = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0)
                && measure == EDiscrete;
        const auto hasScattered = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component == 1)
                && measure == ESolidAngle;

        const auto costheta_i = Frame::cosTheta(bRec.wi);
        const auto costheta_o = Frame::cosTheta(bRec.wo);
        if ((!hasDirect && !hasScattered) || costheta_o <= 0 || costheta_i <= 0)
            return Spectrum(0.0f);
        
        Assert(!!bRec.pltCtx);
        
        const auto m00 = m_specularReflectance->eval(bRec.its);
        const auto a = gfs.alpha(costheta_i, costheta_o);
        
        if (!hasDirect && a.average()==1)
            return Spectrum(.0f);
        
        if (hasDirect) {
            if (std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
                return a * m00 *
                    fresnelConductorApprox(costheta_i, m_eta, m_k);
        }
        if (hasScattered) {
            const auto m = normalize(bRec.wo+bRec.wi);
            return (Spectrum(1.f)-a) * costheta_o * m00 * 
                    gfs.envelopeScattered(*bRec.pltCtx, bRec.wo + bRec.wi) *
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

        const auto costheta_i = Frame::cosTheta(bRec.wi);
        const auto costheta_o = Frame::cosTheta(bRec.wo);
        if ((!hasDirect && !hasScattered) || costheta_o <= 0 || costheta_i <= 0)
            return Spectrum(0.0f);
        
        Assert(bRec.mode==EImportance && rpp.isValid());
        Assert(!!bRec.pltCtx);
        
        const auto rcp_max_lambda = 1.f/Spectrum::lambdas().max();
        const auto m00 = m_specularReflectance->eval(bRec.its);
        const auto a = gfs.alpha(costheta_i, costheta_o);
        const auto phii = bRec.wi.z<1.f ? std::atan2(bRec.wi.y,bRec.wi.x) : .0f;
        const auto phio = bRec.wo.z<1.f ? std::atan2(bRec.wo.y,bRec.wo.x) : .0f;
        
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
        
        const auto in = rpp.spectrum();
        Spectrum result = Spectrum(.0f);
        for (std::size_t idx=0; idx<rpp.size(); ++idx) {
            // Mueller Fresnel pBSDF
            const auto eta = std::complex<Float>(m_eta[idx],m_k[idx]);

            const auto M = hasDirect ? 
                           MuellerFresnelRConductor(costheta_i, eta) :
                           MuellerFresnelRspm1(costheta_i, 1.f, phii, phio, eta);
                        //    MuellerFresnelRConductor(dot(normalize(bRec.wo+bRec.wi),bRec.wi), eta);

            // Diffraction
            auto D = 1.f;
            if (!hasDirect) {
                const auto lambda = Spectrum::lambdas()[idx] * rcp_max_lambda;
                const auto k = Spectrum::ks()[idx];
                const auto h = bRec.wo + bRec.wi;
                const auto invSigma = rpp.invTheta(k,bRec.pltCtx->sigma_zz * 1e+6f);
                D = sqr(1/lambda) * gfs.diffract(invSigma, Q,Qt, k*h);
            }

            auto L = m00[idx] * ((Matrix4x4)M * D*rpp.S(idx));
            L *= hasDirect ? a[idx] : costheta_o * (1-a[idx]);
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

        const auto costheta_i = Frame::cosTheta(bRec.wi);
        const auto costheta_o = Frame::cosTheta(bRec.wo);
        if (costheta_o <= 0 || costheta_i <= 0)
            return .0f;

        Assert(!!bRec.pltCtx);

        Float probDirect = hasDirect ? 1.0f : 0.0f;
        if (hasDirect && hasScattered) {
            const auto a = gfs.alpha(costheta_i, costheta_o);
            probDirect = a.average();
        }

        if (hasDirect && measure == EDiscrete) {
            /* Check if the provided direction pair matches an ideal
               specular reflection; tolerate some roundoff errors */
            if (std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
                return probDirect;
        } else if (hasScattered && measure == ESolidAngle && probDirect<1) {
            return /* costheta_o */ 
                gfs.scatteredPdf(*bRec.pltCtx, bRec.wi, bRec.wo) * (1-probDirect);
        }

        return 0.0f;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        const auto hasDirect = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0);
        const auto hasScattered = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component == 1);

        const auto costheta_i = Frame::cosTheta(bRec.wi);
        const auto costheta_o = Frame::cosTheta(bRec.wo);
        if ((!hasScattered && !hasDirect) || costheta_i <= 0)
            return Spectrum(0.0f);

        Assert(!!bRec.pltCtx);
        
        const auto m00 = m_specularReflectance->eval(bRec.its);
        const auto a = gfs.alpha(costheta_i, costheta_o);

        bRec.eta = 1.0f;
        if (hasScattered && hasDirect) {
            const auto probDirect = a.average();

            if (probDirect==1 || sample.x < probDirect) {
                bRec.sampledComponent = 0;
                bRec.sampledType = EDirectReflection;
                bRec.wo = reflect(bRec.wi);

                return 1.f/probDirect * a * m00 *
                    fresnelConductorApprox(costheta_i, m_eta, m_k);
            } else {
                bRec.sampledComponent = 1;
                bRec.sampledType = EScatteredReflection;
                bRec.wo = gfs.sampleScattered(*bRec.pltCtx, bRec.wi, *bRec.sampler);

                const auto pdf = (1-probDirect) * gfs.scatteredPdf(*bRec.pltCtx, bRec.wi, bRec.wo);
                if (pdf <= RCPOVERFLOW)
                    return Spectrum(.0f);

                const auto m = normalize(bRec.wo+bRec.wi);
                return (1.f/pdf) * costheta_o * (Spectrum(1.f)-a) * m00 *
                    fresnelConductorApprox(dot(m,bRec.wi), m_eta, m_k);
            }
        } else if (hasDirect) {
            bRec.sampledComponent = 0;
            bRec.sampledType = EDirectReflection;
            bRec.wo = reflect(bRec.wi);
            return a * m00 *
                fresnelConductorApprox(costheta_i, m_eta, m_k);
        } else {
            bRec.sampledComponent = 1;
            bRec.sampledType = EScatteredReflection;
            bRec.wo = gfs.sampleScattered(*bRec.pltCtx, bRec.wi, *bRec.sampler);

            const auto pdf = gfs.scatteredPdf(*bRec.pltCtx, bRec.wi, bRec.wo);
            if (pdf <= RCPOVERFLOW)
                return Spectrum(.0f);

            const auto m = normalize(bRec.wo+bRec.wi);
            return (1.f/pdf) * costheta_o * (Spectrum(1.f)-a) * m00 *
                fresnelConductorApprox(dot(m,bRec.wi), m_eta, m_k);
        }
    }
    
    Spectrum getSpecularReflectance(const Intersection &its) const override {
        return m_specularReflectance->eval(its);
    }
    virtual void getRefractiveIndex(const Intersection &its, Spectrum &n, Spectrum &k) const override {
        n = m_eta;
        k = m_k;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "Conductor[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "  eta = " << m_eta.toString() << "," << endl
            << "  k = " << m_k.toString() << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_specularReflectance;
    Spectrum m_eta, m_k;
    gaussian_fractal_surface gfs;
};


MTS_IMPLEMENT_CLASS_S(Conductor, false, BSDF)
MTS_EXPORT_PLUGIN(Conductor, "Rough conductor BRDF");
MTS_NAMESPACE_END
