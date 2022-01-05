
/*
    Copyright, PLT authors
*/

#pragma once

#include "mitsuba/core/constants.h"
#include <complex>
#include <mitsuba/core/math.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/util.h>

#include <cmath>
#include <cassert>

MTS_NAMESPACE_BEGIN

namespace birefringence {
    
using vec2 = Vector2;
using vec3 = Vector3;

// y is up, incident ray is assumed to reside on the yz-plane

// Normal modes
inline float qo(float eo, float K) {
    const float k = 1;
        return sqrt(sqr(eo)*sqr(k) - sqr(K));
}
inline vec2 qe(float eo, float ee, vec3 A, float K) {
    const float e = sqr(ee)-sqr(eo);
    const float k = 1;
    const float a = A.x;
    const float b = A.y;
    const float c = A.z;
    const float G = sqr(b)*e+sqr(eo);
    
    const float qep = (eo*math::safe_sqrt(sqr(K)*(sqr(a)*e-sqr(ee)) + sqr(k)*sqr(ee)*G) + c*b*K*e)/G;
    const float qem = (-eo*math::safe_sqrt(sqr(K)*(sqr(a)*e-sqr(ee)) + sqr(k)*sqr(ee)*G) + c*b*K*e)/G;

    return vec2(qep, qem);
}
// Electric fields
inline vec3 E_ep(float eo, float ee, vec3 A, float K) {
    const float k = 1;
    const float a = A.x;
    const float b = A.y;
    const float c = A.z;
    const float q = qe(eo,ee,A,K).x;
    
    const vec3 E_e = vec3(a*sqr(k)*sqr(eo), 
                          K*c*q + b*(sqr(eo)*sqr(k) - sqr(q)), 
                          K*b*q + c*(sqr(eo)*sqr(k) - sqr(K)));
    return normalize(E_e);
}
inline vec3 E_em(float eo, float ee, vec3 A, float K) {
    const float k = 1;
    const float a = A.x;
    const float b = A.y;
    const float c = A.z;
    const float q = qe(eo,ee,A,K).y;
    
    const vec3 E_e = vec3(a*sqr(k)*sqr(eo), 
                          K*c*q + b*(sqr(eo)*sqr(k) - sqr(q)), 
                          K*b*q + c*(sqr(eo)*sqr(k) - sqr(K)));
    return normalize(E_e);
}
inline vec3 E_op(float eo, vec3 A, float K) {
    // const float k = 1;
    const float a = A.x;
    const float b = A.y;
    const float c = A.z;
    const float q = qo(eo,K);
    
    const vec3 E_o = vec3(-b*K-c*q, a*K, a*q);
    return normalize(E_o);
}
inline vec3 E_om(float eo, vec3 A, float K) {
    // const float k = 1;
    const float a = A.x;
    const float b = A.y;
    const float c = A.z;
    const float q = -qo(eo,K);
    
    const vec3 E_o = vec3(-b*K-c*q, a*K, a*q);
    return normalize(E_o);
}
// Poynting vector
inline vec3 Poynting(float q, vec3 E, float K) {
        return normalize(
                vec3(E.x*(K*E.z - q*E.y),
                         K*E.z*E.y + q*(sqr(E.x) + sqr(E.z)),
                         -q*E.z*E.y - K*(sqr(E.x) + sqr(E.y))));
}

// Fresnel coefficients
inline void fresnel_iso_aniso(float cos_theta, float sin_theta, float ei, float eo, float ee, vec3 A,
                              float& rss, float& rsp, float& tso, float& tse, float& rps, float& rpp, float& tpo, float& tpe) {
    const float k = 1;
    const float K = k*ei*sin_theta;
    const float qi = -qo(ei,K);
    const vec3 E2om = E_om(eo,A,K);
    const vec3 E2em = E_em(eo,ee,A,K);
    const float q2om = -qo(eo,K);
    const float q2em = qe(eo,ee,A,K).y;

    const float A_1 = -E2em.x*E2om.y*K*cos_theta;
    const float A_2 = -E2em.y*E2om.x*K*cos_theta;
    const float C_1 = qi*cos_theta - K*sin_theta;
    const float C_2 = E2em.x*E2om.z*(C_1 + q2om*cos_theta) - A_1;
    const float C_3 = E2em.z*E2om.x*(C_1 + q2em*cos_theta) - A_2;

    const float Ns = C_2*(qi + q2em) - C_3*(qi + q2om);

    rps = (2*E2em.x*E2om.x*cos_theta*C_1*(q2om - q2em)) / Ns;
    rpp = (-E2em.x*E2om.z*(C_1 - q2om*cos_theta)*(qi + q2em) + E2em.z*E2om.x*(C_1 - q2em*cos_theta)*(qi + q2om) - A_1*(q2em + qi) + A_2*(q2om + qi)) / Ns;
    tpo = -(2*E2em.x*cos_theta*(q2em + qi)*C_1) / Ns;
    tpe = (2*E2om.x*cos_theta*(q2om + qi)*C_1) / Ns;

    rss = (C_2*(qi - q2em) + C_3*(q2om - qi)) / Ns;
    rsp = (2*qi*(E2em.z*E2om.y*K - E2em.y*E2om.z*K + E2em.z*E2om.z*(q2om - q2em))) / Ns;
    tso = -(2*qi*(E2em.y*K*cos_theta + E2em.z*C_1 + E2em.z*q2em*cos_theta)) / Ns;
    tse = (2*qi*(E2om.y*K*cos_theta + E2om.z*C_1 + E2om.z*q2om*cos_theta)) / Ns;

}
inline void fresnel_aniso_iso(float cos_phi, float sin_phi, float ei, float eo, float ee, vec3 A,
                              float& roo, float& roe, float& tos, float& top, float& reo, float& ree, float& tes, float& tep) {
    const float eo2 = ei;
    const float k = 1;
    const float K = k * ei * sin_phi;
    const float q2om = -qo(eo2,K);
    const float qop  =  qo(eo,K);
    const vec2 _qe = qe(eo,ee,A,K);
    const float qep = _qe.x;
    const float qem = _qe.y;
    const vec3 Eop = E_op(eo,A,K);
    const vec3 Eep = E_ep(eo,ee,A,K);
    const vec3 Eom = E_om(eo,A,K);
    const vec3 Eem = E_em(eo,ee,A,K);

    const float B_1 = -Eep.x*K*cos_phi*(q2om - qep);
    const float B_2 = -Eom.x*K*cos_phi*(q2om + qop);
    const float B_3 = -Eop.x*K*cos_phi*(qop - q2om);
    const float D_1 = K*sin_phi - q2om*cos_phi;
    const float D_2 = Eep.z*(D_1 + qep*cos_phi);
    const float D_3 = Eem.z*(D_1 + qem*cos_phi);
    const float D_4 = Eop.z*(D_1 + qop*cos_phi);

    const float N2 = (Eep.x*(qep - q2om)*D_4 + Eop.x*(q2om - qop)*D_2 + Eop.y*B_1 + Eep.y*B_3);

    roo = -(Eep.x*Eom.z*(q2om - qep)*(qop*cos_phi - D_1) + Eom.x*(q2om + qop)*D_2 + Eom.y*B_1 - Eep.y*B_2) / N2;
    roe = -(-Eom.x*Eop.z*(qop*(qop*cos_phi + K*sin_phi) + q2om*D_1) + Eom.z*Eop.x*(qop*(qop*cos_phi - K*sin_phi) + q2om*D_1) + Eop.y*B_2 + Eom.y*B_3) / N2;
    tos = (Eep.x*qep*roe - Eom.x*qop + Eop.x*qop*roo) / q2om;
    top = -(Eom.z*qop - Eom.y*K - (Eep.y*K + Eep.z*qep)*roe - (Eop.y*K + Eop.z*qop)*roo) / D_1;

    reo = (Eem.x*(qem - q2om)*(D_2 + Eep.y*K*cos_phi) + Eep.x*(q2om - qep)*(D_3 + Eem.y*K*cos_phi)) / N2;
    ree = -(Eem.x*(qem - q2om)*D_4 - Eem.x*Eop.y*K*cos_phi*(q2om - qem) + Eem.y*B_3 + Eop.x*(q2om - qop)*D_3) / N2;
    tes = (Eem.x*qem + Eep.x*qep*ree + Eop.x*qop*reo) / q2om;
    tep = -(Eem.z + Eep.z*ree + Eop.z*reo) / cos_phi;
}

// Comptutes the four vectors in an anistrotpic slab
void vectors_in_slab(vec3 I, float ei, float eo, float ee, vec3 A, 
                     vec3& Io, vec3& Io2, vec3& Ie, vec3& Ie2, float& e_eff, float& e_eff2) {
    const float K = ei * I.z;
    // const float qop = qo(eo,K);
    const float qom = -qo(eo,K);
    const float qep = qe(eo,ee,A,K).x;
    const float qem = qe(eo,ee,A,K).y;
    // const vec3 Eop = E_op(eo,A,K);
    const vec3 Eom = E_om(eo,A,K);
    const vec3 Eep = E_ep(eo,ee,A,K);
    const vec3 Eem = E_em(eo,ee,A,K);

    Io = Poynting(qom,Eom,K);
    Io2 = Io;
    Io2.y *= -1.f;
    Ie = Poynting(qem,Eem,K);
    Ie2 = Poynting(qep,Eep,K);

    // Effective refractive indices
    const vec3 We =  normalize(vec3(0,qem,-K));
    const vec3 We2 = normalize(vec3(0,qep,-K));
    e_eff = K / math::safe_sqrt(1.f - sqr(We.y));
    e_eff2 = K / math::safe_sqrt(1.f - sqr(We2.y));
}

}

MTS_NAMESPACE_END
