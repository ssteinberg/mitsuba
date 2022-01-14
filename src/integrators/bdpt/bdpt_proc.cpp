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

#include <mitsuba/core/statistics.h>
#include <mitsuba/core/sfcurve.h>
#include <mitsuba/bidir/util.h>
#include <mitsuba/plt/plt.hpp>
#include "bdpt_proc.h"
#include "mitsuba/bidir/common.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                         Worker implementation                        */
/* ==================================================================== */

class BDPTRenderer : public WorkProcessor {
public:
    BDPTRenderer(const BDPTConfiguration &config) : m_config(config) { }

    BDPTRenderer(Stream *stream, InstanceManager *manager)
        : WorkProcessor(stream, manager), m_config(stream) { }

    virtual ~BDPTRenderer() { }

    void serialize(Stream *stream, InstanceManager *manager) const {
        m_config.serialize(stream);
    }

    ref<WorkUnit> createWorkUnit() const {
        return new RectangularWorkUnit();
    }

    ref<WorkResult> createWorkResult() const {
        return new BDPTWorkResult(m_config, m_rfilter.get(),
            Vector2i(m_config.blockSize));
    }

    void prepare() {
        Scene *scene = static_cast<Scene *>(getResource("scene"));
        m_scene = new Scene(scene);
        m_sampler = static_cast<Sampler *>(getResource("sampler"));
        m_sensor = static_cast<Sensor *>(getResource("sensor"));
        m_rfilter = m_sensor->getFilm()->getReconstructionFilter();
        m_scene->removeSensor(scene->getSensor());
        m_scene->addSensor(m_sensor);
        m_scene->setSensor(m_sensor);
        m_scene->setSampler(m_sampler);
        m_scene->wakeup(NULL, m_resources);
        m_scene->initializeBidirectional();
    }

    void process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop) {
        const RectangularWorkUnit *rect = static_cast<const RectangularWorkUnit *>(workUnit);
        BDPTWorkResult *result = static_cast<BDPTWorkResult *>(workResult);
        bool needsTimeSample = m_sensor->needsTimeSample();
        Float time = m_sensor->getShutterOpen();

        result->setOffset(rect->getOffset());
        result->setSize(rect->getSize());
        result->clear();
        m_hilbertCurve.initialize(TVector2<uint8_t>(rect->getSize()));

        #if defined(MTS_DEBUG_FP)
            enableFPExceptions();
        #endif

        Path emitterSubpath;
        Path sensorSubpath;

        /* Determine the necessary random walk depths based on properties of
           the endpoints */
        int emitterDepth = m_config.maxDepth,
            sensorDepth = m_config.maxDepth;

        /* Go one extra step if the sensor can be intersected */
        if (!m_scene->hasDegenerateSensor() && emitterDepth != -1)
            ++emitterDepth;

        /* Go one extra step if there are emitters that can be intersected */
        if (!m_scene->hasDegenerateEmitters() && sensorDepth != -1)
            ++sensorDepth;

        for (size_t i=0; i<m_hilbertCurve.getPointCount(); ++i) {
            Point2i offset = Point2i(m_hilbertCurve[i]) + Vector2i(rect->getOffset());
            m_sampler->generate(offset);

            for (size_t j = 0; j<m_sampler->getSampleCount(); j++) {
                if (stop)
                    break;

                if (needsTimeSample)
                    time = m_sensor->sampleTime(m_sampler->next1D());

                /* Start new emitter and sensor subpaths */
                emitterSubpath.initialize(m_scene, time, EImportance, m_pool);
                sensorSubpath.initialize(m_scene, time, ERadiance, m_pool);

                /* Perform a random walk using alternating steps on each path */
                Path::alternatingRandomWalkFromPixel(m_scene, m_sampler, 
                    m_config.pltCtx,
                    emitterSubpath, emitterDepth, sensorSubpath,
                    sensorDepth, offset, m_config.rrDepth, m_pool);

                evaluate(result, emitterSubpath, sensorSubpath);

                emitterSubpath.release(m_pool);
                sensorSubpath.release(m_pool);

                m_sampler->advance();
            }
        }

        #if defined(MTS_DEBUG_FP)
            disableFPExceptions();
        #endif

        /* Make sure that there were no memory leaks */
        Assert(m_pool.unused());
    }

    /// Evaluate the contributions of the given eye and light paths
    void evaluate(BDPTWorkResult *wr,
            Path &emitterSubpath, Path &sensorSubpath) {
        Point2 initialSamplePos = sensorSubpath.vertex(1)->getSamplePosition();
        const Scene *scene = m_scene;
        PathVertex tempEndpoint, tempSample;
        PathEdge tempEdge, connectionEdge;

        BDAssert(!m_config.sampleDirect && "Not implemented");
 
        int maxS = (int)emitterSubpath.vertexCount()-1;
        std::vector<RadiancePacket> rps;
        std::vector<Spectrum> sweights;
        rps.reserve(emitterSubpath.vertexCount()-1);
        {
            RadiancePacket rp{};
            for (int s=0; s<maxS; ++s) {
                PathVertex
                    *vs = emitterSubpath.vertex(s),
                    *vsPred = emitterSubpath.vertexOrNull(s-1),
                    *vsNext = emitterSubpath.vertexOrNull(s+1);
                
                if (!vs->update(scene, vsPred, vsNext, 
                                &rp, m_config.pltCtx, EImportance, nullptr, vs->measure)) {
                    maxS = s;
                    break;
                }
                rps.push_back(rp);
            }

            sweights.resize(maxS+1);
            sweights[0] = Spectrum(1.f);
            for (int i=1; i<=maxS; ++i)
                sweights[i] = sweights[i-1] *
                    emitterSubpath.vertex(i-1)->weight[EImportance] *
                    emitterSubpath.vertex(i-1)->rrWeight *
                    emitterSubpath.edge(i-1)->weight[EImportance];
        }

        Spectrum sampleValue(0.0f);
        for (int s = maxS; s >= 0; --s) {
            PathVertex
                *vs = emitterSubpath.vertex(s),
                *vsPred = emitterSubpath.vertexOrNull(s-1);
            PathEdge *vsEdge = emitterSubpath.edgeOrNull(s-1);

            /* Determine the range of sensor vertices to be traversed,
               while respecting the specified maximum path length */
            int minT = std::max(2-s, m_config.lightImage ? 0 : 2),
                maxT = (int)sensorSubpath.vertexCount()-1;
            if (m_config.maxDepth != -1)
                maxT = std::min(maxT, m_config.maxDepth + 1 - s);

            for (int t = maxT; t >= minT; --t) {
                PathVertex *vt = sensorSubpath.vertex(t);
                PathVertex *vtNext = sensorSubpath.vertexOrNull(t-1);
                PathEdge *vtEdge = sensorSubpath.edgeOrNull(t-1);

                /* Stores the pixel position associated with this sample */
                Point2 samplePos = initialSamplePos;
                Spectrum value = sweights[s];
                if (value.isZero())
                    continue;

                RestoreMeasureHelper rmhvs(vs), rmhvt(vt);

                /* Account for the terms of the measurement contribution
                   function that are coupled to the connection endpoints */
                if (vs->isEmitterSupernode()) {
                    /* If possible, convert 'vt' into an emitter sample */
                    if (!vt->cast(scene, PathVertex::EEmitterSample) || 
                        vt->isDegenerate())
                        continue;
                } else if (vt->isSensorSupernode()) {
                    /* If possible, convert 'vs' into an sensor sample */
                    if (!vs->cast(scene, PathVertex::ESensorSample) || 
                        vs->isDegenerate())
                        continue;
                    /* Make note of the changed pixel sample position */
                    if (!vs->getSamplePosition(vsPred, samplePos))
                        continue;
                } else {
                    /* Can't connect degenerate endpoints */
                    if (vs->isDegenerate() || vt->isDegenerate())
                        continue;
                }

                /* Determine the pixel sample position when necessary */
                if (vt->isSensorSample() && !vt->getSamplePosition(vs, samplePos))
                    continue;
                
                // Propagate coherence information and update all weights
                RadiancePacket rp = s>0 ? rps[s-1] : RadiancePacket{};
                if (!vs->update(scene, vsPred, vt, 
                                &rp, m_config.pltCtx, EImportance, &value))
                    continue;

                /* Attempt to connect the two endpoints, which could result in
                   the creation of additional vertices (index-matched boundaries etc.) */
                int interactions = m_config.maxDepth - s - t + 1;
                if (!connectionEdge.pathConnectAndCollapse(
                        scene, vsEdge, vs, vt, m_config.pltCtx, rp, vtEdge, interactions))
                    continue;
                    
                if (!vt->update(scene, vs, vtNext, 
                                &rp, m_config.pltCtx, ERadiance, &value))
                    continue;

                /* Temporarily force vertex measure to EArea. Needed to
                   handle BSDFs with diffuse + specular components */
                if (!vs->isEmitterSupernode())
                    vs->measure = EArea;
                if (!vt->isSensorSupernode())
                    vt->measure = EArea;

                // Finish evaluating the chain
                for (int tt = t-1; tt>=std::max(1,minT); --tt) {
                    PathVertex 
                        *vtt = sensorSubpath.vertex(tt),
                        *vttPred = sensorSubpath.vertexOrNull(tt+1),
                        *vttNext = sensorSubpath.vertexOrNull(tt-1);
                
                    vtt->update(scene, vttPred, vttNext, 
                                &rp, m_config.pltCtx, ERadiance, nullptr, vtt->measure);
                    
                    value *=
                        vtt->weight[ERadiance] *
                        vtt->rrWeight *
                        sensorSubpath.edge(tt)->weight[ERadiance];
                    if (value.isZero())
                        break;
                }
                value *=
                    sensorSubpath.vertex(0)->weight[ERadiance] *
                    sensorSubpath.vertex(0)->rrWeight *
                    sensorSubpath.edge(0)->weight[ERadiance];
                if (value.isZero())
                    continue;

                value *= connectionEdge.evalCached(vs, vt,
                                                   PathEdge::EGeneralizedGeometricTerm);
                /* Compute the multiple importance sampling weight */
                value *= Path::miWeight(scene, m_config.pltCtx,
                    emitterSubpath, &connectionEdge, sensorSubpath, 
                    s, t, m_config.sampleDirect, m_config.lightImage);

                if (t >= 2)
                    sampleValue += value;
                else
                    wr->putLightSample(samplePos, value);
            }
        }
        wr->putSample(initialSamplePos, sampleValue);
    }

    ref<WorkProcessor> clone() const {
        return new BDPTRenderer(m_config);
    }

    MTS_DECLARE_CLASS()
private:
    ref<Scene> m_scene;
    ref<Sensor> m_sensor;
    ref<Sampler> m_sampler;
    ref<ReconstructionFilter> m_rfilter;
    MemoryPool m_pool;
    BDPTConfiguration m_config;
    HilbertCurve2D<uint8_t> m_hilbertCurve;
};


/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

BDPTProcess::BDPTProcess(const RenderJob *parent, RenderQueue *queue,
        const BDPTConfiguration &config) :
    BlockedRenderProcess(parent, queue, config.blockSize), m_config(config) {
    m_refreshTimer = new Timer();
}

ref<WorkProcessor> BDPTProcess::createWorkProcessor() const {
    return new BDPTRenderer(m_config);
}

void BDPTProcess::develop() {
    if (!m_config.lightImage)
        return;
    LockGuard lock(m_resultMutex);
    const ImageBlock *lightImage = m_result->getLightImage();
    m_film->setBitmap(m_result->getImageBlock()->getBitmap());
    m_film->addBitmap(lightImage->getBitmap(), 1.0f / m_config.sampleCount);
    m_refreshTimer->reset();
    m_queue->signalRefresh(m_parent);
}

void BDPTProcess::processResult(const WorkResult *wr, bool cancelled) {
    if (cancelled)
        return;
    const BDPTWorkResult *result = static_cast<const BDPTWorkResult *>(wr);
    ImageBlock *block = const_cast<ImageBlock *>(result->getImageBlock());
    LockGuard lock(m_resultMutex);
    m_progress->update(++m_resultCount);
    if (m_config.lightImage) {
        const ImageBlock *lightImage = m_result->getLightImage();
        m_result->put(result);
        if (m_parent->isInteractive()) {
            /* Modify the finished image block so that it includes the light image contributions,
               which creates a more intuitive preview of the rendering process. This is
               not 100% correct but doesn't matter, as the shown image will be properly re-developed
               every 2 seconds and once more when the rendering process finishes */

            Float invSampleCount = 1.0f / m_config.sampleCount;
            const Bitmap *sourceBitmap = lightImage->getBitmap();
            Bitmap *destBitmap = block->getBitmap();
            int borderSize = block->getBorderSize();
            Point2i offset = block->getOffset();
            Vector2i size = block->getSize();

            for (int y=0; y<size.y; ++y) {
                const Float *source = sourceBitmap->getFloatData()
                    + (offset.x + (y+offset.y) * sourceBitmap->getWidth()) * SPECTRUM_SAMPLES;
                Float *dest = destBitmap->getFloatData()
                    + (borderSize + (y + borderSize) * destBitmap->getWidth()) * (SPECTRUM_SAMPLES + 2);

                for (int x=0; x<size.x; ++x) {
                    Float weight = dest[SPECTRUM_SAMPLES + 1] * invSampleCount;
                    for (int k=0; k<SPECTRUM_SAMPLES; ++k)
                        *dest++ += *source++ * weight;
                    dest += 2;
                }
            }
        }
    }

    m_film->put(block);

    /* Re-develop the entire image every two seconds if partial results are
       visible (e.g. in a graphical user interface). This only applies when
       there is a light image. */
    bool developFilm = m_config.lightImage &&
        (m_parent->isInteractive() && m_refreshTimer->getMilliseconds() > 2000);

    m_queue->signalWorkEnd(m_parent, result->getImageBlock(), false);

    if (developFilm)
        develop();
}

void BDPTProcess::bindResource(const std::string &name, int id) {
    BlockedRenderProcess::bindResource(name, id);
    if (name == "sensor" && m_config.lightImage) {
        /* If needed, allocate memory for the light image */
        m_result = new BDPTWorkResult(m_config, NULL, m_film->getCropSize());
        m_result->clear();
    }
}

MTS_IMPLEMENT_CLASS_S(BDPTRenderer, false, WorkProcessor)
MTS_IMPLEMENT_CLASS(BDPTProcess, false, BlockedRenderProcess)
MTS_NAMESPACE_END
