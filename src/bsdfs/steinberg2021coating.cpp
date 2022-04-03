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
#include <mitsuba/core/warp.h>
#include <mitsuba/core/fresolver.h>
#include "ior.h"
#include "microfacet.h"

MTS_NAMESPACE_BEGIN

/*! \plugin{coating}{Smooth dielectric coating}
 * \order{10}
 * \icon{bsdf_coating}
 *
 * \parameters{
 *     \parameter{intIOR}{\Float\Or\String}{Interior index of refraction specified
 *      numerically or using a known material name. \default{\texttt{bk7} / 1.5046}}
 *     \parameter{extIOR}{\Float\Or\String}{Exterior index of refraction specified
 *      numerically or using a known material name. \default{\texttt{air} / 1.000277}}
 *     \parameter{thickness}{\Float}{Denotes the thickness of the layer (to
 *      model absorption --- should be specified in inverse units of \code{sigmaA})\default{1}}
 *     \parameter{sigmaA}{\Spectrum\Or\Texture}{The absorption coefficient of the
 *      coating layer. \default{0, i.e. there is no absorption}}
 *     \parameter{specular\showbreak Reflectance}{\Spectrum\Or\Texture}{Optional
 *         factor that can be used to modulate the specular reflection component. Note
 *         that for physical realism, this parameter should never be touched. \default{1.0}}
 *     \parameter{\Unnamed}{\BSDF}{A nested BSDF model that should be coated.}
 * }
 *
 * \renderings{
 *     \rendering{Rough copper}
 *         {bsdf_coating_uncoated}
 *     \rendering{The same material coated with a single layer of
 *         clear varnish (see \lstref{coating-roughcopper})}
 *         {bsdf_coating_roughconductor}
 * }
 *
 * This plugin implements a smooth dielectric coating (e.g. a layer of varnish)
 * in the style of the paper ``Arbitrarily Layered Micro-Facet Surfaces'' by
 * Weidlich and Wilkie \cite{Weidlich2007Arbitrarily}. Any BSDF in Mitsuba
 * can be coated using this plugin, and multiple coating layers can even
 * be applied in sequence. This allows designing interesting custom materials
 * like car paint or glazed metal foil. The coating layer can optionally be
 * tinted (i.e. filled with an absorbing medium), in which case this model also
 * accounts for the directionally dependent absorption within the layer.
 *
 * Note that the plugin discards illumination that undergoes internal
 * reflection within the coating. This can lead to a noticeable energy
 * loss for materials that reflect much of their energy near or below the critical
 * angle (i.e. diffuse or very rough materials).
 * Therefore, users are discouraged to use this plugin to coat smooth
 * diffuse materials, since there is a separately available plugin
 * named \pluginref{plastic}, which covers the same case and does not
 * suffer from energy loss.\newpage
 *
 * \renderings{
 *     \smallrendering{$\code{thickness}=0$}{bsdf_coating_0}
 *     \smallrendering{$\code{thickness}=1$}{bsdf_coating_1}
 *     \smallrendering{$\code{thickness}=5$}{bsdf_coating_5}
 *     \smallrendering{$\code{thickness}=15$}{bsdf_coating_15}
 *     \caption{The effect of the layer thickness parameter on
 *        a tinted coating ($\code{sigmaT}=(0.1, 0.2, 0.5)$)}
 * }
 *
 * \vspace{4mm}
 *
 * \begin{xml}[caption=Rough copper coated with a transparent layer of
 *     varnish, label=lst:coating-roughcopper]
 * <bsdf type="coating">
 *     <float name="intIOR" value="1.7"/>
 *
 *     <bsdf type="roughconductor">
 *         <string name="material" value="Cu"/>
 *         <float name="alpha" value="0.1"/>
 *     </bsdf>
 * </bsdf>
 * \end{xml}
 *
 * \renderings{
 *     \rendering{Coated rough copper with a bump map applied on top}{bsdf_coating_coatedbump}
 *     \rendering{Bump mapped rough copper with a coating on top}{bsdf_coating_bumpcoating}
 *     \caption{Some interesting materials can be created simply by applying
 *     Mitsuba's material modifiers in different orders.}
 * }
 *
 * \subsubsection*{Technical details}
 * Evaluating the internal component of this model entails refracting the
 * incident and exitant rays through the dielectric interface, followed by
 * querying the nested material with this modified direction pair. The result
 * is attenuated by the two Fresnel transmittances and the absorption, if
 * any.
 */
class SmoothCoating : public BSDF {
public:
    SmoothCoating(const Properties &props)
            : BSDF(props) {
        m_specularReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("specularReflectance", Spectrum(1.0f)));

        std::string materialName = props.getString("material", "Cu");

        Spectrum intEta, intK;
        if (boost::to_lower_copy(materialName) == "none") {
            intEta = Spectrum(0.0f);
            intK = Spectrum(1.0f);
        } else {
            ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();
            intEta.fromContinuousSpectrum(InterpolatedSpectrum(
                fResolver->resolve("data/ior/" + materialName + ".eta.spd")));
            intK.fromContinuousSpectrum(InterpolatedSpectrum(
                fResolver->resolve("data/ior/" + materialName + ".k.spd")));
        }
        m_coatingEta = lookupIOR(props, "coatingIOR", "bk7");
        m_eta = props.getSpectrum("eta", intEta) / m_coatingEta;
        m_k   = props.getSpectrum("k", intK) / m_coatingEta;

        m_thickness = props.getFloat("thickness");
        m_sigma = props.getFloat("sigma", 0.5);
        
        m_specularSamplingWeight = .1f;
    }

    SmoothCoating(Stream *stream, InstanceManager *manager)
            : BSDF(stream, manager) {
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_thickness = stream->readFloat();
        m_eta = Spectrum(stream);
        m_coatingEta = stream->readFloat();
        m_k = Spectrum(stream);
        m_sigma = stream->readFloat();
        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        manager->serialize(stream, m_specularReflectance.get());
        stream->writeFloat(m_thickness);
        m_eta.serialize(stream);
        stream->writeFloat(m_coatingEta);
        m_k.serialize(stream);
        stream->writeFloat(m_sigma);
    }

    void configure() {
        unsigned int extraFlags = 0;
        if (!m_specularReflectance->isConstant())
            extraFlags |= ESpatiallyVarying;
        
        m_components.clear();
        m_components.push_back(EDirectReflection | EFrontSide | ESpatiallyVarying);
        m_components.push_back(EScatteredReflection | EFrontSide | ESpatiallyVarying | extraFlags);

        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);

        BSDF::configure();
    }

    /// Reflection in local coordinates
    inline Vector reflect(const Vector &wi) const {
        return Vector(-wi.x, -wi.y, wi.z);
    }
    /// Helper function: reflect \c wi with respect to a given surface normal
    inline Vector reflect(const Vector &wi, const Normal &m) const {
        return 2 * dot(wi, m) * Vector(m) - wi;
    }
    
    auto phase_shift_term(const Vector3 &Phi) const {
        Spectrum ret;
        for (int i=0; i<SPECTRUM_SAMPLES; i++) 
            ret[i] = cos(Phi.z * m_thickness * Spectrum::ks()[i]);
        return ret;
    }
    
    Spectrum PSD(const Vector3 &wi, const Vector3 &wo, const Vector3 &Phi) const {
        MicrofacetDistribution distr(
            MicrofacetDistribution::EBeckmann,
            m_sigma*sqrt(2),
            m_sigma*sqrt(2),
            true
        );
        
        auto H = normalize(Phi);
        H *= math::signum(Frame::cosTheta(H));

        return Spectrum(distr.G(wi, wo, H) * distr.eval(H));
    }
    
    Spectrum envelope(const Vector3 &wi, const Vector3 &wo, const Intersection &its, Float sigma2) const {
        auto spec = Spectrum(.0f);
        Float Fi = fresnelDielectricExt(Frame::cosTheta(wi), m_coatingEta);
        if (std::abs(dot(reflect(wi), wo)-1) < DeltaEpsilon)
            spec = m_specularReflectance->eval(its) * Fi;
        
        const Vector Phi = wo+wi;
        const Vector H = normalize(Phi);
        const auto c = phase_shift_term(Phi);

        const Spectrum F = fresnelConductorExact(dot(wi, H), m_eta, m_k);
        const auto refl  = F * PSD(wi, wo, Phi) / (4.0f * dot(wo, H));
        
        const auto Z = BSDF::getFrame(its).toWorld({0,0,2*m_thickness});
        const auto mutual_coh = std::exp(-.5f * dot(Z,Z) * sigma2);
        
        auto result = .5f*(spec + refl + mutual_coh*c*refl);
        result.clampNegative();
        
        return result;
    }
    
    Spectrum envelope(const BSDFSamplingRecord &bRec, Float &eta, EMeasure measure) const {
        bool hasSpecular   = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0)
                && measure == EDiscrete;
        bool hasInterf = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component == 1)
                && measure == ESolidAngle;
        
        if ((!hasSpecular && !hasInterf)
            || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(0.0f);
        
        const auto sigma2 = bRec.pltCtx->sigma2_min_um;
        return envelope(bRec.wi,bRec.wo,bRec.its,sigma2);
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, Float &eta, 
                  RadiancePacket &rpp, EMeasure measure) const { 
        bool hasSpecular   = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0)
                && measure == EDiscrete;
        bool hasInterf = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component == 1)
                && measure == ESolidAngle;
    
        auto wi = bRec.wi;
        
        if ((!hasSpecular && !hasInterf)
            || Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(wi) <= 0)
            return Spectrum(0.0f);
        
        auto spec = Spectrum(.0f);
        Float Fi = fresnelDielectricExt(Frame::cosTheta(wi), m_coatingEta);
        if (std::abs(dot(reflect(wi), bRec.wo)-1) < DeltaEpsilon)
            spec = m_specularReflectance->eval(bRec.its) * Fi;
        
        const Vector Phi = bRec.wo+wi;
        const Vector H = normalize(Phi);
        const auto c = phase_shift_term(Phi);

        const Spectrum F = fresnelConductorExact(dot(wi, H), m_eta, m_k);
        const auto refl  = F * PSD(wi, bRec.wo, Phi) / (4.0f * dot(bRec.wo, H));
        
        const auto Z = BSDF::getFrame(bRec.its).toWorld({0,0,2*m_thickness*1e-6f});
        
        const auto in = rpp.spectrum();
        Spectrum result = Spectrum(.0f);
        for (std::size_t idx=0; idx<rpp.size(); ++idx) {
            const auto k = Spectrum::ks()[idx];
            const auto mutual_coh = rpp.mutualCoherence(k, Z, bRec.pltCtx->sigma_zz * 1e+6f);   // in micron

            auto L = std::max(.0f, 
                              .5f*(spec[idx] + refl[idx] + mutual_coh*c[idx]*refl[idx])) * rpp.S(idx);
            if (in[idx]>RCPOVERFLOW)
                result[idx] = L[0] / in[idx];
            rpp.L(idx) = L;
        }
        
        return result;
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        bool hasSpecular   = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0);
        bool hasInterf = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component == 1);

        if (Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
            return 0.0f;

        Float probSpecular = hasSpecular ? 1.0f : 0.0f;
        if (hasSpecular && hasInterf) {
            Float Fi = fresnelDielectricExt(Frame::cosTheta(bRec.wi), m_coatingEta);
            probSpecular = (Fi*m_specularSamplingWeight) /
                (Fi*m_specularSamplingWeight +
                (1-Fi) * (1-m_specularSamplingWeight));
        }

        if (hasSpecular && measure == EDiscrete) {
            /* Check if the provided direction pair matches an ideal
               specular reflection; tolerate some roundoff errors */
            if (std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
                return probSpecular;
        } else if (hasInterf && measure == ESolidAngle) {
            Vector Phi = bRec.wo+bRec.wi;
            Vector H = normalize(Phi);
            
            const auto dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, H));
            H *= math::signum(Frame::cosTheta(H));
            
            MicrofacetDistribution distr(
                MicrofacetDistribution::EBeckmann,
                m_sigma*sqrt(2),
                m_sigma*sqrt(2),
                true
            );
            return distr.pdf(math::signum(Frame::cosTheta(bRec.wi)) * bRec.wi, H) * dwh_dwo * (1-probSpecular);
        }
        return 0.0f;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &_sample) const {
        bool hasSpecular = (bRec.typeMask & EDirectReflection)
                && (bRec.component == -1 || bRec.component == 0);
        bool hasInterf = (bRec.typeMask & EScatteredReflection)
                && (bRec.component == -1 || bRec.component == 1);

        if ((!hasInterf && !hasInterf) || Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(0.0f);

        const auto sigma2 = bRec.pltCtx->sigma2_min_um;
        Float Fi = fresnelDielectricExt(Frame::cosTheta(bRec.wi), m_coatingEta);

        MicrofacetDistribution distr(
            MicrofacetDistribution::EBeckmann,
            m_sigma*sqrt(2),
            m_sigma*sqrt(2),
            true
        );

        Float pdf{};
        bRec.eta = 1.0f;
        if (hasInterf && hasSpecular) {
            Float probSpecular = (Fi*m_specularSamplingWeight) /
                (Fi*m_specularSamplingWeight +
                (1-Fi) * (1-m_specularSamplingWeight));

            /* Importance sample wrt. the Fresnel reflectance */
            if (_sample.x < probSpecular) {
                bRec.sampledComponent = 0;
                bRec.sampledType = EDirectReflection;
                bRec.wo = reflect(bRec.wi);

                pdf = probSpecular;
                return m_specularReflectance->eval(bRec.its)
                    * Fi / probSpecular;
            } else {
                const auto sample = Point2((_sample.x - probSpecular) / (1 - probSpecular),_sample.y);
                Float microfacetPDF;
                const Normal m = distr.sample(math::signum(Frame::cosTheta(bRec.wi)) * bRec.wi, sample, microfacetPDF);

                bRec.sampledComponent = 1;
                bRec.sampledType = EScatteredReflection;
                bRec.wo = reflect(bRec.wi, m);
                
                pdf = microfacetPDF;
                if (!pdf)
                    return Spectrum(.0f);
                return envelope(bRec.wi,bRec.wo,bRec.its,sigma2) / pdf;
            }
        } else if (hasSpecular) {
            bRec.sampledComponent = 0;
            bRec.sampledType = EDirectReflection;
            bRec.wo = reflect(bRec.wi);
            return m_specularReflectance->eval(bRec.its) * Fi;
        } else {
            Float microfacetPDF;
            const Normal m = distr.sample(math::signum(Frame::cosTheta(bRec.wi)) * bRec.wi, _sample, microfacetPDF);

            bRec.sampledComponent = 1;
            bRec.sampledType = EScatteredReflection;
            bRec.wo = reflect(bRec.wi, m);

            pdf = microfacetPDF;
            if (!pdf)
                return Spectrum(.0f);
            return envelope(bRec.wi,bRec.wo,bRec.its,sigma2);///pdf;
        }
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "SmoothCoating[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  thickness = " << m_thickness << "," << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    ref<Texture> m_specularReflectance;
    Float m_thickness, m_specularSamplingWeight, m_sigma, m_coatingEta;
    Spectrum m_eta;
    Spectrum m_k;
};


MTS_IMPLEMENT_CLASS_S(SmoothCoating, false, BSDF)
MTS_EXPORT_PLUGIN(SmoothCoating, "Smooth dielectric coating");
MTS_NAMESPACE_END
