
/*
    Copyright, PLT authors
*/

#pragma once

#include "mitsuba/core/vector.h"
#include <mitsuba/core/frame.h>
#include <mitsuba/core/matrix.h>

MTS_NAMESPACE_BEGIN

// Mueller rotation matrix between reference frames
inline Matrix4x4 MuellerT(const Frame &f1, const Frame &f2) {
    auto p = f2.s*(1 - dot(f1.n,f2.s));
    Float cost{};
    if (dot(p,p)>1e-5) {
        p = normalize(p);
        cost = dot(p,f1.s);
    }
    else {
        p = f2.t*(1 - dot(f1.n,f2.t));
        p = normalize(p);
        cost = dot(p,f1.t);
    }
    
    const auto sint2 = 1-cost*cost;
    const auto sint  = sqrt(sint2);
    const auto rx    = 2*sint*cost;
    const auto ry    = 1-2*sint2;
    
    Matrix4x4 T{ Float(0) };
    T.m[0][0] = T.m[3][3] = 1;
    T.m[1][1] = T.m[2][2] = rx;
    T.m[1][2] = -ry;
    T.m[2][1] = ry;
    return T;
}

MTS_NAMESPACE_END
