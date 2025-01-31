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

/*! \plugin{normalmap}{Normal map modifier}
 * \order{13}
 * \icon{bsdf_bumpmap}
 *
 * \parameters{
 *     \parameter{\Unnamed}{\Texture}{
 *       The color values of this texture specify the perturbed
 *       normals relative in the local surface coordinate system.
 *     }
 *     \parameter{\Unnamed}{\BSDF}{A BSDF model that should
 *     be affected by the normal map}
 * }
 *
 * This plugin is conceptually similar to the \pluginref{bumpmap} plugin
 * but uses a normal map instead of a bump map. A normal map is a RGB texture, whose color channels
 * encode the XYZ coordinates of the desired surface normals.
 * These are specified \emph{relative} to the local shading frame,
 * which means that a normal map with a value of $(0,0,1)$ everywhere
 * causes no changes to the surface.
 * To turn the 3D normal directions into (nonnegative) color values
 * suitable for this plugin, the
 * mapping $x \mapsto (x+1)/2$ must be applied to each component.
 */
class NormalMap : public BSDF {
public:
    NormalMap(const Properties &props) : BSDF(props) { }

    NormalMap(Stream *stream, InstanceManager *manager)
            : BSDF(stream, manager) {
        m_nested = static_cast<BSDF *>(manager->getInstance(stream));
        m_normals = static_cast<Texture *>(manager->getInstance(stream));
        configure();
    }

    void configure() {
        if (!m_nested)
            Log(EError, "A child BSDF instance is required");
        if (!m_normals)
            Log(EError, "A normal map texture must be specified");

        m_components.clear();
        for (int i=0; i<m_nested->getComponentCount(); ++i)
            m_components.push_back(m_nested->getType(i) | ESpatiallyVarying | EAnisotropic);

        BSDF::configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        manager->serialize(stream, m_nested.get());
        manager->serialize(stream, m_normals.get());
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
            if (m_normals != NULL)
                Log(EError, "Only a single normal texture can be specified!");
            const Properties &props = child->getProperties();
            if (props.getPluginName() == "bitmap" && !props.hasProperty("gamma"))
                Log(EError, "When using a bitmap texture as a normal map, please explicitly specify "
                        "the 'gamma' parameter of the bitmap plugin. In most cases the following is the correct choice: "
                        "<float name=\"gamma\" value=\"1.0\"/>");
            m_normals = static_cast<Texture *>(child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    Frame getFrame(const Intersection &its) const {
        Frame result;
        Normal n;

        m_normals->eval(its, false).toLinearRGB(n.x, n.y, n.z);
        for (int i=0; i<3; ++i)
            n[i] = 2 * n[i] - 1;

        Frame frame = BSDF::getFrame(its);
        result.n = normalize(frame.toWorld(n));

        result.s = normalize(its.dpdu - result.n
            * dot(result.n, its.dpdu));

        result.t = cross(result.n, result.s);

        return result;
    }

    void getFrameDerivative(const Intersection &its, Frame &du, Frame &dv) const {
        Vector n;

        m_normals->eval(its, false).toLinearRGB(n.x, n.y, n.z);
        for (int i=0; i<3; ++i)
            n[i] = 2 * n[i] - 1;

        Spectrum dn[2];
        Vector dndu, dndv;
        m_normals->evalGradient(its, dn);
        Spectrum(2*dn[0]).toLinearRGB(dndu.x, dndu.y, dndu.z);
        Spectrum(2*dn[1]).toLinearRGB(dndv.x, dndv.y, dndv.z);

        Frame base_du, base_dv;
        Frame base = BSDF::getFrame(its);
        BSDF::getFrameDerivative(its, base_du, base_dv);

        Vector worldN = base.toWorld(n);

        Float invLength_n = 1/worldN.length();
        worldN *= invLength_n;

        du.n = invLength_n * (base.toWorld(dndu) + base_du.toWorld(n));
        dv.n = invLength_n * (base.toWorld(dndv) + base_dv.toWorld(n));
        du.n -= dot(du.n, worldN) * worldN;
        dv.n -= dot(dv.n, worldN) * worldN;

        Vector s = its.dpdu - worldN * dot(worldN, its.dpdu);
        Float invLen_s = 1.0f / s.length();
        s *= invLen_s;

        du.s = invLen_s * (-du.n * dot(worldN, its.dpdu) - worldN * dot(du.n, its.dpdu));
        dv.s = invLen_s * (-dv.n * dot(worldN, its.dpdu) - worldN * dot(dv.n, its.dpdu));

        du.s -= s * dot(du.s, s);
        dv.s -= s * dot(dv.s, s);

        du.t = cross(du.n, s) + cross(worldN, du.s);
        dv.t = cross(dv.n, s) + cross(worldN, dv.s);
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
        oss << "NormalMap[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  normals = " << indent(m_normals->toString()) << endl
            << "  nested = " << indent(m_nested->toString()) << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    ref<Texture> m_normals;
    ref<BSDF> m_nested;
};

MTS_IMPLEMENT_CLASS_S(NormalMap, false, BSDF)
MTS_EXPORT_PLUGIN(NormalMap, "Normal map modifier");
MTS_NAMESPACE_END
