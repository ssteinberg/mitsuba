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

#include <mitsuba/render/scene.h>
#include <mitsuba/hw/basicshader.h>

MTS_NAMESPACE_BEGIN

/*! \plugin{bumpmap}{Bump map modifier}
 * \order{12}
 * \icon{bsdf_bumpmap}
 *
 * \parameters{
 *     \parameter{\Unnamed}{\Texture}{
 *       The luminance of this texture specifies the amount of
 *       displacement. The implementation ignores any constant
 *       offset---only changes in the luminance matter.
 *     }
 *     \parameter{\Unnamed}{\BSDF}{A BSDF model that should
 *     be affected by the bump map}
 * }
 * \renderings{
 *     \rendering{Bump map based on tileable diagonal lines}{bsdf_bumpmap_1}
 *     \rendering{An irregular bump map}{bsdf_bumpmap_2}
 * }
 *
 * Bump mapping \cite{Blinn1978Simulation} is a simple technique for cheaply
 * adding surface detail to a rendering. This is done by perturbing the
 * shading coordinate frame based on a displacement height field provided
 * as a texture. This method can lend objects a highly realistic and detailed
 * appearance (e.g. wrinkled or covered by scratches and other imperfections)
 * without requiring any changes to the input geometry.
 *
 * The implementation in Mitsuba uses the common approach of ignoring
 * the usually negligible texture-space derivative of the base mesh
 * surface normal. As side effect of this decision, it is invariant
 * to constant offsets in the height field texture---only variations in
 * its luminance cause changes to the shading frame.
 *
 * Note that the magnitude of the height field variations influences
 * the strength of the displacement. If desired, the \pluginref{scale}
 * texture plugin can be used to magnify or reduce the effect of a
 * bump map texture.
 * \begin{xml}[caption=A rough metal model with a scaled image-based bump map]
 * <bsdf type="bumpmap">
 *     <!-- The bump map is applied to a rough metal BRDF -->
 *     <bsdf type="roughconductor"/>
 *
 *     <texture type="scale">
 *         <!-- The scale of the displacement gets multiplied by 10x -->
 *         <float name="scale" value="10"/>
 *
 *         <texture type="bitmap">
 *             <string name="filename" value="bumpmap.png"/>
 *         </texture>
 *     </texture>
 * </bsdf>
 * \end{xml}
 */
class BumpMap : public BSDF {
public:
    BumpMap(const Properties &props) : BSDF(props) { }

    BumpMap(Stream *stream, InstanceManager *manager)
            : BSDF(stream, manager) {
        m_nested = static_cast<BSDF *>(manager->getInstance(stream));
        m_displacement = static_cast<Texture *>(manager->getInstance(stream));
        configure();
    }

    void configure() {
        if (!m_nested)
            Log(EError, "A child BSDF instance is required");
        if (!m_displacement)
            Log(EError, "A displacement texture must be specified");

        m_components.clear();
        for (int i=0; i<m_nested->getComponentCount(); ++i)
            m_components.push_back(m_nested->getType(i) | ESpatiallyVarying | EAnisotropic);

        BSDF::configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        manager->serialize(stream, m_nested.get());
        manager->serialize(stream, m_displacement.get());
    }

    Spectrum getDiffuseReflectance(const Intersection &its) const {
        return m_nested->getDiffuseReflectance(its);
    }

    Spectrum getSpecularReflectance(const Intersection &its) const {
        return m_nested->getSpecularReflectance(its);
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(BSDF))) {
            if (m_nested != NULL)
                Log(EError, "Only a single nested BSDF can be added!");
            m_nested = static_cast<BSDF *>(child);
        } else if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
            if (m_displacement != NULL)
                Log(EError, "Only a single displacement texture can be specified!");
            const Properties &props = child->getProperties();
            if (props.getPluginName() == "bitmap" && !props.hasProperty("gamma"))
                Log(EError, "When using a bitmap texture as a bump map, please explicitly specify "
                        "the 'gamma' parameter of the bitmap plugin. In most cases the following is the correct choice: "
                        "<float name=\"gamma\" value=\"1.0\"/>");
            m_displacement = static_cast<Texture *>(child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    Frame getFrame(const Intersection &its) const {
        Spectrum grad[2];
        m_displacement->evalGradient(its, grad);

        Float dDispDu = grad[0].getLuminance();
        Float dDispDv = grad[1].getLuminance();

        /* Build a perturbed frame -- ignores the usually
           negligible normal derivative term */
        Vector dpdu = its.dpdu + its.shFrame.n * (
                dDispDu - dot(its.shFrame.n, its.dpdu));
        Vector dpdv = its.dpdv + its.shFrame.n * (
                dDispDv - dot(its.shFrame.n, its.dpdv));

        Frame result;
        result.n = normalize(cross(dpdu, dpdv));
        result.s = normalize(dpdu - result.n
            * dot(result.n, dpdu));
        result.t = cross(result.n, result.s);

        if (dot(result.n, its.geoFrame.n) < 0)
            result.n *= -1;

        return result;
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, Float &eta,
                  RadiancePacket &rpp, EMeasure measure) const { 
        const Intersection& its = bRec.its;
        Intersection perturbed(its);
        perturbed.shFrame = getFrame(its);

        BSDFSamplingRecord perturbedQuery(perturbed,
            perturbed.toLocal(its.toWorld(bRec.wi)),
            perturbed.toLocal(its.toWorld(bRec.wo)), bRec.mode);
        if (Frame::cosTheta(bRec.wo) * Frame::cosTheta(perturbedQuery.wo) <= 0)
            return Spectrum(0.0f);
        perturbedQuery.sampler = bRec.sampler;
        perturbedQuery.typeMask = bRec.typeMask;
        perturbedQuery.component = bRec.component;
        perturbedQuery.pltCtx = bRec.pltCtx;
        return m_nested->eval(perturbedQuery, eta, rpp, measure);
    }

    Spectrum envelope(const BSDFSamplingRecord &bRec, Float &eta, EMeasure measure) const {
        const Intersection& its = bRec.its;
        Intersection perturbed(its);
        perturbed.shFrame = getFrame(its);

        BSDFSamplingRecord perturbedQuery(perturbed,
            perturbed.toLocal(its.toWorld(bRec.wi)),
            perturbed.toLocal(its.toWorld(bRec.wo)), bRec.mode);
        if (Frame::cosTheta(bRec.wo) * Frame::cosTheta(perturbedQuery.wo) <= 0)
            return Spectrum(0.0f);
        perturbedQuery.sampler = bRec.sampler;
        perturbedQuery.typeMask = bRec.typeMask;
        perturbedQuery.component = bRec.component;
        perturbedQuery.pltCtx = bRec.pltCtx;
        return m_nested->envelope(perturbedQuery, eta, measure);
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        const Intersection& its = bRec.its;
        Intersection perturbed(its);
        perturbed.shFrame = getFrame(its);

        BSDFSamplingRecord perturbedQuery(perturbed,
            perturbed.toLocal(its.toWorld(bRec.wi)),
            perturbed.toLocal(its.toWorld(bRec.wo)), bRec.mode);
        if (Frame::cosTheta(bRec.wo) * Frame::cosTheta(perturbedQuery.wo) <= 0)
            return 0;
        perturbedQuery.mode = bRec.mode;
        perturbedQuery.sampler = bRec.sampler;
        perturbedQuery.typeMask = bRec.typeMask;
        perturbedQuery.component = bRec.component;
        perturbedQuery.pltCtx = bRec.pltCtx;
        return m_nested->pdf(perturbedQuery, measure);
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        const Intersection& its = bRec.its;
        Intersection perturbed(its);
        perturbed.shFrame = getFrame(its);

        BSDFSamplingRecord perturbedQuery(perturbed, bRec.sampler, bRec.mode);
        perturbedQuery.wi = perturbed.toLocal(its.toWorld(bRec.wi));
        perturbedQuery.sampler = bRec.sampler;
        perturbedQuery.typeMask = bRec.typeMask;
        perturbedQuery.component = bRec.component;
        perturbedQuery.pltCtx = bRec.pltCtx;
        Spectrum result = m_nested->sample(perturbedQuery, sample);
        if (!result.isZero()) {
            bRec.sampledComponent = perturbedQuery.sampledComponent;
            bRec.sampledType = perturbedQuery.sampledType;
            bRec.wo = its.toLocal(perturbed.toWorld(perturbedQuery.wo));
            bRec.eta = perturbedQuery.eta;
            if (Frame::cosTheta(bRec.wo) * Frame::cosTheta(perturbedQuery.wo) <= 0)
                return Spectrum(0.0f);
        }
        return result;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "BumpMap[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  displacement = " << indent(m_displacement->toString()) << endl
            << "  nested = " << indent(m_nested->toString()) << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    ref<Texture> m_displacement;
    ref<BSDF> m_nested;
};

MTS_IMPLEMENT_CLASS_S(BumpMap, false, BSDF)
MTS_EXPORT_PLUGIN(BumpMap, "Bump map modifier");
MTS_NAMESPACE_END
