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

#include <mitsuba/core/warp.h>
#include "mitsuba/core/constants.h"
#include "mitsuba/core/fwd.h"
#include "mitsuba/core/math.h"
#include "mitsuba/render/sampler.h"
#include <cmath>

MTS_NAMESPACE_BEGIN

namespace warp {

Vector squareToUniformSphere(const Point2 &sample) {
    Float z = 1.0f - 2.0f * sample.y;
    Float r = math::safe_sqrt(1.0f - z*z);
    Float sinPhi, cosPhi;
    math::sincos(2.0f * M_PI * sample.x, &sinPhi, &cosPhi);
    return Vector(r * cosPhi, r * sinPhi, z);
}

Vector squareToUniformHemisphere(const Point2 &sample) {
    Float z = sample.x;
    Float tmp = math::safe_sqrt(1.0f - z*z);

    Float sinPhi, cosPhi;
    math::sincos(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);

    return Vector(cosPhi * tmp, sinPhi * tmp, z);
}

Vector squareToCosineHemisphere(const Point2 &sample) {
    const auto p = squareToUniformDiskConcentric(sample);
    return diskToCosineHemisphere(p);
}

Vector squareToUniformCone(Float cosCutoff, const Point2 &sample) {
    Float cosTheta = (1-sample.x) + sample.x * cosCutoff;
    Float sinTheta = math::safe_sqrt(1.0f - cosTheta * cosTheta);

    Float sinPhi, cosPhi;
    math::sincos(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);

    return Vector(cosPhi * sinTheta,
        sinPhi * sinTheta, cosTheta);
}

Point2 squareToUniformDisk(const Point2 &sample) {
    Float r = std::sqrt(sample.x);
    Float sinPhi, cosPhi;
    math::sincos(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);

    return Point2(
        cosPhi * r,
        sinPhi * r
    );
}

Point2 squareToUniformTriangle(const Point2 &sample) {
    Float a = math::safe_sqrt(1.0f - sample.x);
    return Point2(1 - a, a * sample.y);
}

Point2 squareToUniformDiskConcentric(const Point2 &sample) {
    Float r1 = 2.0f*sample.x - 1.0f;
    Float r2 = 2.0f*sample.y - 1.0f;

    /* Modified concencric map code with less branching (by Dave Cline), see
       http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html */
    Float phi, r;
    if (r1 == 0 && r2 == 0) {
        r = phi = 0;
    } else if (r1*r1 > r2*r2) {
        r = r1;
        phi = (M_PI/4.0f) * (r2/r1);
    } else {
        r = r2;
        phi = (M_PI/2.0f) - (r1/r2) * (M_PI/4.0f);
    }

    Float cosPhi, sinPhi;
    math::sincos(phi, &sinPhi, &cosPhi);

    return Point2(r * cosPhi, r * sinPhi);
}

Point2 uniformDiskToSquareConcentric(const Point2 &p) {
    Float r   = std::sqrt(p.x * p.x + p.y * p.y),
          phi = std::atan2(p.y, p.x),
          a, b;

    if (phi < -M_PI/4) {
        /* in range [-pi/4,7pi/4] */
        phi += 2*M_PI;
    }

    if (phi < M_PI/4) { /* region 1 */
        a = r;
        b = phi * a / (M_PI/4);
    } else if (phi < 3*M_PI/4) { /* region 2 */
        b = r;
        a = -(phi - M_PI/2) * b / (M_PI/4);
    } else if (phi < 5*M_PI/4) { /* region 3 */
        a = -r;
        b = (phi - M_PI) * a / (M_PI/4);
    } else { /* region 4 */
        b = -r;
        a = -(phi - 3*M_PI/2) * b / (M_PI/4);
    }

    return Point2(0.5f * (a+1), 0.5f * (b+1));
}

Point2 squareToStdNormal(const Point2 &sample) {
    Float r   = std::sqrt(-2 * std::log(1-sample.x)),
          phi = 2 * M_PI * sample.y;
    Point2 result;
    math::sincos(phi, &result.y, &result.x);
    return result * r;
}

Float squareToStdNormalPdf(const Point2 &pos) {
    return INV_TWOPI * std::exp(-(pos.x*pos.x + pos.y*pos.y)/2.0f);
}

Vector squareToTruncatedGaussian(Float stddev, const Point2 &mean, Sampler &sampler) {
    constexpr auto attempts = 1000ull;
    for (auto i=0ull; i<attempts; ++i) {
        const auto& p = squareToStdNormal(sampler.next2D())*stddev + mean;
        const auto l = sqr(p.x)+sqr(p.y);
        if (l<=1) {
            auto z = std::sqrt(1-l);
            if (EXPECT_NOT_TAKEN(z < 1e-10f))
                z = 1e-10f;
            return Vector{ p.x,p.y,z };
        }
    }

    assert(false);  // Rejection sampling failed
    return Vector{0,0,0};
}

Float squareToTruncatedGaussianPdf(Float stddev, const Point2 &mean, const Vector3 &pos) {
    const auto& p = Point2{ pos.x-mean.x,pos.y-mean.y } / stddev;

    const auto ymax = math::safe_sqrt(1-sqr(pos.x));
    const auto cdfxmin = math::normalCDF((-1    - mean.x) / stddev);
    const auto cdfxmax = math::normalCDF(( 1    - mean.x) / stddev);
    const auto cdfymin = math::normalCDF((-ymax - mean.y) / stddev);
    const auto cdfymax = math::normalCDF(( ymax - mean.y) / stddev);

    const auto r = (cdfxmax-cdfxmin) * (cdfymax-cdfymin) * sqr(stddev);
    return r>RCPOVERFLOW ? std::exp(-(sqr(p.x) + sqr(p.y))/2) * INV_TWOPI / r : .0f;
}

static Float intervalToTent(Float sample) {
    Float sign;

    if (sample < 0.5f) {
        sign = 1;
        sample *= 2;
    } else {
        sign = -1;
        sample = 2 * (sample - 0.5f);
    }

    return sign * (1 - std::sqrt(sample));
}

Point2 squareToTent(const Point2 &sample) {
    return Point2(
        intervalToTent(sample.x),
        intervalToTent(sample.y)
    );
}

Float intervalToNonuniformTent(Float a, Float b, Float c, Float sample) {
    Float factor;

    if (sample * (c-a) < b-a) {
        factor = a-b;
        sample *= (a-c)/(a-b);
    } else {
        factor = c-b;
        sample = (a-c)/(b-c) * (sample - (a-b)/(a-c));
    }

    return b + factor * (1-math::safe_sqrt(sample));
}

};

MTS_NAMESPACE_END
