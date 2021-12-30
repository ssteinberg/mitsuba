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
#include "microfacet.h"
#include "ior.h"
#include "mitsuba/core/fwd.h"
#include "mitsuba/core/math.h"
#include "mitsuba/core/vector.h"
#include "mitsuba/core/warp.h"
#include "mitsuba/plt/mueller.h"
#include "mitsuba/render/sampler.h"
#include "mitsuba/render/shape.h"

#include <boost/math/distributions/normal.hpp>

MTS_NAMESPACE_BEGIN

/*!\plugin{roughconductor}{Rough conductor material}
 * \order{7}
 * \icon{bsdf_roughconductor}
 * \parameters{
 *     \parameter{distribution}{\String}{
 *          Specifies the type of microfacet normal distribution
 *          used to model the surface roughness.
 *          \vspace{-1mm}
 *       \begin{enumerate}[(i)]
 *           \item \code{beckmann}: Physically-based distribution derived from
 *               Gaussian random surfaces. This is the default.\vspace{-1.5mm}
 *           \item \code{ggx}: The GGX \cite{Walter07Microfacet} distribution (also known as
 *               Trowbridge-Reitz \cite{Trowbridge19975Average} distribution)
 *               was designed to better approximate the long tails observed in measurements
 *               of ground surfaces, which are not modeled by the Beckmann distribution.
 *           \vspace{-1.5mm}
 *           \item \code{phong}: Anisotropic Phong distribution by
 *              Ashikhmin and Shirley \cite{Ashikhmin2005Anisotropic}.
 *              In most cases, the \code{ggx} and \code{beckmann} distributions
 *              should be preferred, since they provide better importance sampling
 *              and accurate shadowing/masking computations.
 *              \vspace{-4mm}
 *       \end{enumerate}
 *     }
 *     \parameter{alpha, alphaU, alphaV}{\Float\Or\Texture}{
 *         Specifies the roughness of the unresolved surface micro-geometry
 *         along the tangent and bitangent directions. When the Beckmann
 *         distribution is used, this parameter is equal to the
 *         \emph{root mean square} (RMS) slope of the microfacets.
 *         \code{alpha} is a convenience parameter to initialize both
 *         \code{alphaU} and \code{alphaV} to the same value. \default{0.1}.
 *     }
 *     \parameter{material}{\String}{Name of a material preset, see
 *           \tblref{conductor-iors}.\!\default{\texttt{Cu} / copper}}
 *     \parameter{eta, k}{\Spectrum}{Real and imaginary components of the material's index of
 *             refraction \default{based on the value of \texttt{material}}}
 *     \parameter{extEta}{\Float\Or\String}{
 *           Real-valued index of refraction of the surrounding dielectric,
 *           or a material name of a dielectric \default{\code{air}}
 *     }
 *     \parameter{sampleVisible}{\Boolean}{
 *         Enables a sampling technique proposed by Heitz and D'Eon~\cite{Heitz1014Importance},
 *         which focuses computation on the visible parts of the microfacet normal
 *         distribution, considerably reducing variance in some cases.
 *         \default{\code{true}, i.e. use visible normal sampling}
 *     }
 *     \parameter{specular\showbreak Reflectance}{\Spectrum\Or\Texture}{Optional
 *         factor that can be used to modulate the specular reflection component. Note
 *         that for physical realism, this parameter should never be touched. \default{1.0}}
 * }
 * \vspace{3mm}
 * This plugin implements a realistic microfacet scattering model for rendering
 * rough conducting materials, such as metals. It can be interpreted as a fancy
 * version of the Cook-Torrance model and should be preferred over
 * heuristic models like \pluginref{phong} and \pluginref{ward} if possible.
 * \renderings{
 *     \rendering{Rough copper (Beckmann, $\alpha=0.1$)}
 *         {bsdf_roughconductor_copper.jpg}
 *     \rendering{Vertically brushed aluminium (Anisotropic Phong,
 *         $\alpha_u=0.05,\ \alpha_v=0.3$), see
 *         \lstref{roughconductor-aluminium}}
 *         {bsdf_roughconductor_anisotropic_aluminium.jpg}
 * }
 *
 * Microfacet theory describes rough surfaces as an arrangement of unresolved
 * and ideally specular facets, whose normal directions are given by a
 * specially chosen \emph{microfacet distribution}. By accounting for shadowing
 * and masking effects between these facets, it is possible to reproduce the
 * important off-specular reflections peaks observed in real-world measurements
 * of such materials.
 *
 * This plugin is essentially the ``roughened'' equivalent of the (smooth) plugin
 * \pluginref{conductor}. For very low values of $\alpha$, the two will
 * be identical, though scenes using this plugin will take longer to render
 * due to the additional computational burden of tracking surface roughness.
 *
 * The implementation is based on the paper ``Microfacet Models
 * for Refraction through Rough Surfaces'' by Walter et al.
 * \cite{Walter07Microfacet}. It supports three different types of microfacet
 * distributions and has a texturable roughness parameter.
 * To facilitate the tedious task of specifying spectrally-varying index of
 * refraction information, this plugin can access a set of measured materials
 * for which visible-spectrum information was publicly available
 * (see \tblref{conductor-iors} for the full list).
 * There is also a special material profile named \code{none}, which disables
 * the computation of Fresnel reflectances and produces an idealized
 * 100% reflecting mirror.
 *
 * When no parameters are given, the plugin activates the default settings,
 * which describe copper with a medium amount of roughness modeled using a
 * Beckmann distribution.
 *
 * To get an intuition about the effect of the surface roughness parameter
 * $\alpha$, consider the following approximate classification: a value of
 * $\alpha=0.001-0.01$ corresponds to a material with slight imperfections
 * on an otherwise smooth surface finish, $\alpha=0.1$ is relatively rough,
 * and $\alpha=0.3-0.7$ is \emph{extremely} rough (e.g. an etched or ground
 * finish). Values significantly above that are probably not too realistic.
 * \vspace{4mm}
 * \begin{xml}[caption={A material definition for brushed aluminium}, label=lst:roughconductor-aluminium]
 * <bsdf type="roughconductor">
 *     <string name="material" value="Al"/>
 *     <string name="distribution" value="phong"/>
 *     <float name="alphaU" value="0.05"/>
 *     <float name="alphaV" value="0.3"/>
 * </bsdf>
 * \end{xml}
 *
 * \subsubsection*{Technical details}
 * All microfacet distributions allow the specification of two distinct
 * roughness values along the tangent and bitangent directions. This can be
 * used to provide a material with a ``brushed'' appearance. The alignment
 * of the anisotropy will follow the UV parameterization of the underlying
 * mesh. This means that such an anisotropic material cannot be applied to
 * triangle meshes that are missing texture coordinates.
 *
 * \label{sec:visiblenormal-sampling}
 * Since Mitsuba 0.5.1, this plugin uses a new importance sampling technique
 * contributed by Eric Heitz and Eugene D'Eon, which restricts the sampling
 * domain to the set of visible (unmasked) microfacet normals. The previous
 * approach of sampling all normals is still available and can be enabled
 * by setting \code{sampleVisible} to \code{false}.
 * Note that this new method is only available for the \code{beckmann} and
 * \code{ggx} microfacet distributions. When the \code{phong} distribution
 * is selected, the parameter has no effect.
 *
 * When rendering with the Phong microfacet distribution, a conversion is
 * used to turn the specified Beckmann-equivalent $\alpha$ roughness value
 * into the exponent parameter of this distribution. This is done in a way,
 * such that the same value $\alpha$ will produce a similar appearance across
 * different microfacet distributions.
 *
 * When using this plugin, you should ideally compile Mitsuba with support for
 * spectral rendering to get the most accurate results. While it also works
 * in RGB mode, the computations will be more approximate in nature.
 * Also note that this material is one-sided---that is, observed from the
 * back side, it will be completely black. If this is undesirable,
 * consider using the \pluginref{twosided} BRDF adapter.
 */
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
        unsigned int extraFlagsDirect = 0;
        unsigned int extraFlagsScattered = 0;
        if (!m_q->isConstant() || !m_sigma2->isConstant() || !m_specularReflectance->isConstant())
            extraFlagsScattered |= ESpatiallyVarying;
        if (!m_specularReflectance->isConstant())
            extraFlagsDirect |= ESpatiallyVarying;

        m_components.clear();
        m_components.push_back(EDirectReflection | EFrontSide | extraFlagsDirect);
        m_components.push_back(EScatteredReflection | EFrontSide | EUsesSampler | extraFlagsScattered);

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

    /// Helper function: reflect \c wi with respect to a given surface normal
    inline Vector reflect(const Vector &wi) const {
        const auto m = Vector3{ 0,0,1 };
        return 2 * dot(wi, m) * Vector(m) - wi;
    }
    
    inline auto alpha(const Float costhetai, const Float q) const {
        const auto a = -sqr(2 * costhetai * q) * Spectrum::ks() * Spectrum::ks();
        return a.exp();
    }

    inline auto diffract(const Matrix3x3& invSigma, Float sigma2, const Frame &f, const Vector3 &h) const {
        Matrix3x3 Q = Matrix3x3(f.s,f.t,f.n), Qt;
        Q.transpose(Qt);
        const auto& invTheta = Q*invSigma*Qt;
        const auto& invTheta2x2 = Matrix2x2(invTheta.m[0][0], invTheta.m[0][1],
                                            invTheta.m[0][1], invTheta.m[1][1]);
        
        const Matrix2x2& S = invTheta2x2 + (Float(1)/sigma2) * Matrix2x2(1,0,0,1);
        Matrix2x2 invS;
        S.invert2x2(invS);
        const auto& h2 = Vector2(h.x,h.y);

        return Float(.5)*INV_PI/std::sqrt(S.det()) * std::exp(-Float(.5)*dot(h2,invS*h2));
    }

    auto envelopeScattered(const Intersection &its, const PLTContext &pltCtx, const Vector3 &h) const {
        const auto sigma_min = pltCtx.sigma_zz * 1e+6; // metres to micron
        const auto sigma2 = m_sigma2->eval(its).average();
        const auto w = Float(1) / sigma_min + Float(1) / sigma2;
        
        const auto dist = boost::math::normal{ Float(0), w };
        Spectrum gx, gy;
        for (std::size_t i=0;i<SPECTRUM_SAMPLES;++i) {
            const auto kh = Spectrum::ks()[i]*h;
            gx[i] = boost::math::pdf(dist, kh.x);
            gy[i] = boost::math::pdf(dist, kh.y);
        }
        
        return gx*gy;
    }

    auto sampleScattered(const Intersection &its, const PLTContext &pltCtx, const Vector3 &wi, Sampler &sampler) const {
        const auto sigma_min = pltCtx.sigma_zz * 1e+6; // metres to micron
        const auto sigma2 = m_sigma2->eval(its).average();
        const auto w = Float(1) / sigma_min + Float(1) / sigma2;
        const auto k = Spectrum::ks().average();
        
        return warp::squareToClampedGaussian(std::sqrt(w)/k, -Point2{ wi.x,wi.y }, sampler);
    }

    auto scatteredPdf(const Intersection &its, const PLTContext &pltCtx, const Vector3 &wi, const Vector3 &wo) const {
        const auto sigma_min = pltCtx.sigma_zz * 1e+6; // metres to micron
        const auto sigma2 = m_sigma2->eval(its).average();
        const auto w = Float(1) / sigma_min + Float(1) / sigma2;
        const auto k = Spectrum::ks().average();
        
        return warp::squareToClampedGaussianPdf(std::sqrt(w)/k, -Point2{ wi.x,wi.y }, Point2{ wo.x,wo.y });
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
        const auto costheta_o = Frame::cosTheta(bRec.wo);
        const auto a = alpha(Frame::cosTheta(bRec.wi), q);
        
        if (hasDirect) {
            if (std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
                return a * m00;
            return Spectrum(.0f);
        }
        return costheta_o * sqr(q) * (Spectrum(1.f)-a) * m00 * 
                envelopeScattered(bRec.its, *bRec.pltCtx, bRec.wo + bRec.wi);
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
        
        Assert(bRec.mode==ERadiance && radiancePacket.isValid());
        Assert(!!bRec.pltCtx);
        
        const auto m00 = m_specularReflectance->eval(bRec.its);
        const auto q = m_q->eval(bRec.its).average();
        const auto sigma2 = m_sigma2->eval(bRec.its).average();
        const auto costheta_o = Frame::cosTheta(bRec.wo);
        const auto a = alpha(Frame::cosTheta(bRec.wi), q);
        const auto m = normalize(bRec.wo+bRec.wi);
        
        // Rotate to sp-frame first
        radiancePacket.rotateFrame(bRec.its, Frame::spframe(bRec.wo,Normal{ 0,0,1 }));
        
        const auto& in = radiancePacket.spectrum();
        Spectrum result = Spectrum(.0f);
        for (std::size_t idx=0; idx<radiancePacket.size(); ++idx) {
            // Mueller Fresnel pBSDF
            const auto M = sqr(Spectrum::lambdas().average()/Spectrum::lambdas()[idx]) * 
                           MuellerFresnelRConductor(dot(m,bRec.wi), m_eta[idx]);

            if (hasDirect) {
                if (std::abs(dot(reflect(bRec.wi), bRec.wo)-1) >= DeltaEpsilon)
                    return Spectrum(.0f);
                
                radiancePacket.L(idx) = a[idx] * m00[idx] * ((Matrix4x4)M * radiancePacket.S(idx));
            }
            else {
                const auto k = Spectrum::ks()[idx];
                const auto h = k*(bRec.wo + bRec.wi);
                const auto Dx = diffract(radiancePacket.invThetax(k,bRec.pltCtx->sigma_zz), 
                                         sigma2, bRec.its.shFrame, h);
                const auto Dy = diffract(radiancePacket.invThetay(k,bRec.pltCtx->sigma_zz), 
                                         sigma2, bRec.its.shFrame, h);
                const auto Dc = diffract(radiancePacket.invThetac(k,bRec.pltCtx->sigma_zz), 
                                         sigma2, bRec.its.shFrame, h);
                
                radiancePacket.L(idx) = costheta_o * sqr(q) * (1-a[idx]) * m00[idx] * ((Matrix4x4)M * 
                               (Dx*radiancePacket.Sx(idx) + Dy*radiancePacket.Sy(idx) + Dc*radiancePacket.Sc(idx)));
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
            probDirect = alpha(Frame::cosTheta(bRec.wi), q).average();
        }

        if (hasDirect && measure == EDiscrete) {
            /* Check if the provided direction pair matches an ideal
               specular reflection; tolerate some roundoff errors */
            if (std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
                return probDirect;
        } else if (hasScattered && measure == ESolidAngle) {
            return scatteredPdf(bRec.its, *bRec.pltCtx, bRec.wi, bRec.wo) * (1-probDirect);
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
        const auto a = alpha(Frame::cosTheta(bRec.wi), q);

        bRec.eta = 1.0f;
        if (hasScattered && hasDirect) {
            const auto probDirect = a.average();

            if (sample.x < probDirect) {
                bRec.sampledComponent = 0;
                bRec.sampledType = EDirectReflection;
                bRec.wo = reflect(bRec.wi);

                return Float(1)/probDirect * a * m00;
            } else {
                bRec.sampledComponent = 1;
                bRec.sampledType = EDirectReflection;
                bRec.wo = sampleScattered(bRec.its, *bRec.pltCtx, bRec.wi, *bRec.sampler);

                return sqr(q)/(1-probDirect) * (Spectrum(1.f)-a) * m00;
            }
        } else if (hasDirect) {
            bRec.sampledComponent = 0;
            bRec.sampledType = EDirectReflection;
            bRec.wo = reflect(bRec.wi);
            return a * m00;
        } else {
            bRec.sampledComponent = 1;
            bRec.sampledType = EScatteredReflection;
            bRec.wo = sampleScattered(bRec.its, *bRec.pltCtx, bRec.wi, *bRec.sampler);

            return sqr(q) * (Spectrum(1.f)-a) * m00;
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
