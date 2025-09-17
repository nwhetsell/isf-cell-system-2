/*{
    "CATEGORIES": [
        "Filter",
        "Generator"
    ],
    "CREDIT": "Mykhailo Moroz <https://www.shadertoy.com/user/michael0884>",
    "DESCRIPTION": "Cell system, converted from <https://www.shadertoy.com/view/3tSfRW>",
    "INPUTS": [
        {
            "NAME" : "inputImage",
            "TYPE" : "image"
        },
        {
            "NAME": "inputImageAmount",
            "LABEL": "Input image amount",
            "TYPE": "float",
            "DEFAULT": 0,
            "MIN": 0,
            "MAX": 1
        },
        {
            "NAME": "restart",
            "LABEL": "Restart",
            "TYPE": "event"
        },
        {
            "NAME": "enableMouse",
            "LABEL": "Enable mouse",
            "TYPE": "bool",
            "DEFAULT": false
        },
        {
            "NAME": "mouse",
            "TYPE": "point2D",
            "DEFAULT": [0.5, 0.5],
            "MIN": [0, 0],
            "MAX": [1, 1]
        },
        {
            "NAME": "dt",
            "LABEL": "Simulation speed",
            "TYPE": "float",
            "DEFAULT": 1,
            "MAX": 10,
            "MIN": 0
        },
        {
            "NAME": "distribution_size",
            "LABEL": "Distribution size",
            "TYPE": "float",
            "DEFAULT": 1.7,
            "MAX": 10,
            "MIN": 0
        },
        {
            "NAME": "acceleration",
            "LABEL": "Acceleration",
            "TYPE": "float",
            "DEFAULT": 0,
            "MAX": 1,
            "MIN": 0
        },
        {
            "NAME": "sense_ang",
            "LABEL": "Sensor angle factor",
            "TYPE": "float",
            "DEFAULT": 0.03,
            "MAX": 2,
            "MIN": 0
        },
        {
            "NAME": "sense_dis",
            "LABEL": "Sensor distance",
            "TYPE": "float",
            "DEFAULT": 15,
            "MAX": 20,
            "MIN": 0
        },
        {
            "NAME": "distance_scale",
            "LABEL": "Sensor distance scale",
            "TYPE": "float",
            "DEFAULT": 0,
            "MAX": 1,
            "MIN": 0
        },
        {
            "NAME": "sense_oscil",
            "LABEL": "Sensor turn speed",
            "TYPE": "float",
            "DEFAULT": 0,
            "MAX": 1,
            "MIN": 0
        },
        {
            "NAME": "oscil_scale",
            "LABEL": "Sensor turn speed scale",
            "TYPE": "float",
            "DEFAULT": 1,
            "MAX": 1,
            "MIN": 0
        },
        {
            "NAME": "oscil_pow",
            "LABEL": "Oscillation power",
            "TYPE": "float",
            "DEFAULT": 0,
            "MAX": 10,
            "MIN": -10
        },
        {
            "NAME": "sense_force",
            "LABEL": "Sensor strength",
            "TYPE": "float",
            "DEFAULT": 0.38,
            "MAX": 1,
            "MIN": -1
        },
        {
            "NAME": "force_scale",
            "LABEL": "Sensor force scale",
            "TYPE": "float",
            "DEFAULT": 1,
            "MAX": 2,
            "MIN": 0
        },
        {
            "NAME": "density_normalization_speed",
            "LABEL": "Density normalization speed",
            "TYPE": "float",
            "DEFAULT": 0.13,
            "MAX": 1,
            "MIN": 0
        },
        {
            "NAME": "density_target",
            "LABEL": "Density target",
            "TYPE": "float",
            "DEFAULT": 0.24,
            "MAX": 1,
            "MIN": 0
        },
        {
            "NAME": "vorticity_confinement",
            "LABEL": "Vorticity confinement",
            "TYPE": "float",
            "DEFAULT": 0,
            "MAX": 100,
            "MIN": 0
        },
        {
            "NAME": "massDecayFactor",
            "LABEL": "Mass decay",
            "TYPE": "float",
            "DEFAULT": 1,
            "MAX": 2,
            "MIN": 0
        },
        {
            "NAME": "radius",
            "LABEL": "Smoothing radius",
            "TYPE": "float",
            "DEFAULT": 2,
            "MAX": 10,
            "MIN": 0
        }
    ],
    "ISFVSN": "2",
    "PASSES": [
        {
            "TARGET": "bufferA_positionAndMass",
            "PERSISTENT": true,
            "FLOAT": true
        },
        {
            "TARGET": "bufferA_velocity",
            "PERSISTENT": true,
            "FLOAT": true
        },
        {
            "TARGET": "bufferB",
            "PERSISTENT": true,
            "FLOAT": true
        },
        {
            "TARGET": "bufferC",
            "PERSISTENT": true,
            "FLOAT": true
        },
        {
            "TARGET": "bufferD",
            "PERSISTENT": true,
            "FLOAT": true
        },
        {

        }
    ]
}*/


// Constants and functions from LYGIA <https://github.com/patriciogonzalezvivo/lygia>
#define PI 3.1415926535897932384626433832795


//
// ShaderToy Common
//

// Hash function from <https://www.shadertoy.com/view/4djSRW>, MIT-licensed:
//
// Copyright © 2014 David Hoskins.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the “Software”), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
float hash11(float p)
{
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

#define HALF_SENSOR_COUNT_MINUS_1 12

//useful functions
#define GS(x) exp(-dot(x,x))
#define GSS(x) exp(-dot(x,x))
#define GS0(x) exp(-length(x))
#define Dir(ang) vec2(cos(ang), sin(ang))
#define Rot(ang) mat2(cos(ang), sin(ang), -sin(ang), cos(ang))
#define loop(i,x) for(int i = 0; i < x; i++)
#define range(i,a,b) for(int i = a; i <= b; i++)

//SPH pressure

//density*temp = pressure - i.e. ideal gas law
#define Pressure(rho) 0.5*rho.z


//data packing
#define POST_UNPACK(X) (clamp(X, 0., 1.) * 2. - 1.)
#define PRE_PACK(X) clamp(0.5 * X + 0.5, 0., 1.)


void main()
{
    vec2 position = gl_FragCoord.xy;

    if (PASSINDEX == 0 || PASSINDEX == 1) // ShaderToy Buffer A
    {
        vec2 X = vec2(0);
        vec2 V = vec2(0);
        float M = 0.;

        //basically integral over all updated neighbor distributions
        //that fall inside of this pixel
        //this makes the tracking conservative
        range(i, -2, 2) range(j, -2, 2)
        {
            vec2 tpos = position + vec2(i,j);
            vec2 wrapped_tpos = mod(tpos, RENDERSIZE);
            vec4 data = IMG_PIXEL(bufferA_positionAndMass, wrapped_tpos);

            vec2 X0 = POST_UNPACK(data.xy) + tpos;
           	vec2 V0 = POST_UNPACK(IMG_PIXEL(bufferB, wrapped_tpos).xy);
           	float M0 = data.z;

            X0 += V0*dt; //integrate position

            //particle distribution size
            float K = distribution_size;

            vec4 aabbX = vec4(max(position - 0.5, X0 - K*0.5), min(position + 0.5, X0 + K*0.5)); //overlap aabb
            vec2 center = 0.5*(aabbX.xy + aabbX.zw); //center of mass
            vec2 size = max(aabbX.zw - aabbX.xy, 0.); //only positive

            //the deposited mass into this cell
            float m = M0*size.x*size.y/(K*K);

            //add weighted by mass
            X += center*m;
            V += V0*m;

            //add mass
            M += m;
        }

        //normalization
        if(M != 0.)
        {
            X /= M;
            V /= M;
        }

        //mass renormalization
        float prevM = M;
        M = mix(M, density_target, density_normalization_speed);
        V = V*prevM/M;

        // Mass decay
        M *= massDecayFactor;

        //initial condition
        if(FRAMEINDEX < 1 || restart)
        {
            X = position;
            vec2 dx0 = (position - RENDERSIZE*0.3); vec2 dx1 = (position - RENDERSIZE*0.7);
            V = 0.5*Rot(PI*0.5)*dx0*GS(dx0/30.) - 0.5*Rot(PI*0.5)*dx1*GS(dx1/30.);
            V += 1.0*Dir(2.*PI*hash11(floor(position.x/10.) + RENDERSIZE.x*floor(position.y/20.)));
            M = 0.1 + position.x/RENDERSIZE.x*0.01 + position.y/RENDERSIZE.x*0.01;
        }

        if (PASSINDEX == 0) {
            X = clamp(X - position, vec2(-0.5), vec2(0.5));
            gl_FragColor = vec4(PRE_PACK(X), M, 1.);
        } else {
            gl_FragColor = vec4(PRE_PACK(V), 0., 1.);
        }
    }
    else if (PASSINDEX == 2) // ShaderToy Buffer B
    {
        vec2 wrapped_pos = mod(position, RENDERSIZE);

        vec4 data = IMG_PIXEL(bufferA_positionAndMass, wrapped_pos);
        vec2 X = POST_UNPACK(data.xy) + position;
        vec2 V = POST_UNPACK(IMG_PIXEL(bufferA_velocity, wrapped_pos).xy);
        float M = data.z;

        if(M != 0.) //not vacuum
        {
            //Compute the force
            vec2 F = vec2(0.);
            const vec2 dx = vec2(0, 1);

            //get neighbor data
            wrapped_pos = mod(position + dx.xy, RENDERSIZE);
            vec4 d_u = IMG_PIXEL(bufferA_positionAndMass, wrapped_pos);
            wrapped_pos = mod(position - dx.xy, RENDERSIZE);
            vec4 d_d = IMG_PIXEL(bufferA_positionAndMass, wrapped_pos);
            wrapped_pos = mod(position + dx.yx, RENDERSIZE);
            vec4 d_r = IMG_PIXEL(bufferA_positionAndMass, wrapped_pos);
            wrapped_pos = mod(position - dx.yx, RENDERSIZE);
            vec4 d_l = IMG_PIXEL(bufferA_positionAndMass, wrapped_pos);

            //position deltas
            vec2 p_u = POST_UNPACK(d_u.xy), p_d = POST_UNPACK(d_d.xy);
            vec2 p_r = POST_UNPACK(d_r.xy), p_l = POST_UNPACK(d_l.xy);

            //velocities
            wrapped_pos = mod(position + dx.xy, RENDERSIZE);
            vec2 v_u = POST_UNPACK(IMG_PIXEL(bufferA_velocity, wrapped_pos).xy);
            wrapped_pos = mod(position - dx.xy, RENDERSIZE);
            vec2 v_d = POST_UNPACK(IMG_PIXEL(bufferA_velocity, wrapped_pos).xy);
            wrapped_pos = mod(position + dx.yx, RENDERSIZE);
            vec2 v_r = POST_UNPACK(IMG_PIXEL(bufferA_velocity, wrapped_pos).xy);
            wrapped_pos = mod(position - dx.yx, RENDERSIZE);
            vec2 v_l = POST_UNPACK(IMG_PIXEL(bufferA_velocity, wrapped_pos).xy);



            //pressure gradient
            vec2 p = vec2(Pressure(d_r) - Pressure(d_l),
                          Pressure(d_u) - Pressure(d_d));

            //density gradient
            vec2 dgrad = vec2(d_r.z - d_l.z,
                          d_u.z - d_d.z);

            //velocity operators
            float div = v_r.x - v_l.x + v_u.y - v_d.y;
            float curl = v_r.y - v_l.y - v_u.x + v_d.x;
            //vec2 laplacian =

            F -= 1.0*M*p;


            float ang = atan(V.y, V.x);
            float dang =sense_ang*PI/float(HALF_SENSOR_COUNT_MINUS_1);
            vec2 slimeF = vec2(0.);
            //slime mold sensors
            range(i, -HALF_SENSOR_COUNT_MINUS_1, HALF_SENSOR_COUNT_MINUS_1)
            {
                float cang = ang + float(i) * dang;
            	vec2 dir = (1. + sense_dis*pow(M, distance_scale))*Dir(cang);
            	vec2 sensedPosition = mod(X + dir, RENDERSIZE);
               	vec3 s0 = IMG_NORM_PIXEL(bufferC, sensedPosition / RENDERSIZE).xyz;
       			float fs = pow(s0.z, force_scale);
                float os = oscil_scale*pow(s0.z - M, oscil_pow);
            	slimeF +=  sense_oscil*Rot(os)*s0.xy
                         + sense_force*Dir(ang + sign(float(i))*PI*0.5)*fs;
            }

            //remove acceleration component from slime force and leave rotation only
            slimeF -= dot(slimeF, normalize(V))*normalize(V);
    		F += slimeF/float(2*HALF_SENSOR_COUNT_MINUS_1);

            // TODO
            // if(iMouse.z > 0.)
            // {
            //     vec2 dx= position - iMouse.xy;
            //      F += 0.1*Rot(PI*0.5)*dx*GS(dx/30.);
            // }

            //integrate velocity
            V += Rot(-vorticity_confinement*curl)*F*dt/M;

            //acceleration for fun effects
            V *= 1. + acceleration;

            //velocity limit
            float v = length(V);
            V /= (v > 1.)?v/1.:1.;
        }

        //mass decay
       // M *= 0.999;

        //input
        //if(iMouse.z > 0.)
        //\\	M = mix(M, 0.5, GS((position - iMouse.xy)/13.));
        //else
         //   M = mix(M, 0.5, GS((position - RENDERSIZE*0.5)/13.));

        //save
        gl_FragColor = vec4(PRE_PACK(V), 0., 1.);
    }
    else if (PASSINDEX == 3) // ShaderToy Buffer C
    {
        float rho = 0.001;
        vec2 vel = vec2(0., 0.);

        //compute the smoothed density and velocity
        range(i, -2, 2) range(j, -2, 2)
        {
            vec2 tpos = position + vec2(i,j);
            vec2 wrapped_tpos = mod(tpos, RENDERSIZE);
            vec4 data = IMG_PIXEL(bufferA_positionAndMass, wrapped_tpos);

            vec2 X0 = POST_UNPACK(data.xy) + tpos;
            vec2 V0 = POST_UNPACK(IMG_PIXEL(bufferB, wrapped_tpos).xy);
            float M0 = data.z;
            vec2 dx = X0 - position;

            float K = GS(dx/radius)/(radius);
            rho += M0*K;
            vel += M0*K*V0;
        }

        vel /= rho;

        gl_FragColor = vec4(vel, rho, 1.0);
    }
    else if (PASSINDEX == 4) // ShaderToy Buffer D
    {
        vec2 wrapped_pos = mod(position, RENDERSIZE);
       	vec2 V0 = POST_UNPACK(IMG_PIXEL(bufferB, wrapped_pos).xy);

        wrapped_pos = mod(position - V0 * dt, RENDERSIZE);
        gl_FragColor = IMG_NORM_PIXEL(bufferD, wrapped_pos / RENDERSIZE);
        //initial condition
        if(FRAMEINDEX < 1 || restart)
        {
            gl_FragColor.xy = position/RENDERSIZE;
        }
    }
    else // ShaderToy Image
    {
        vec2 wrapped_pos = mod(position.xy, RENDERSIZE);
        float r = IMG_NORM_PIXEL(bufferA_positionAndMass, wrapped_pos / RENDERSIZE).z;
       	// vec4 c = texture(bufferD, mod(position.xy, RENDERSIZE) / RENDERSIZE);

       	//get neighbor data
        // vec4 d_u = texelFetch(bufferB, ivec2(mod(position + dx.xy, RENDERSIZE)), 0);
        // vec4 d_d = texelFetch(bufferB, ivec2(mod(position - dx.xy, RENDERSIZE)), 0);
        // vec4 d_r = texelFetch(bufferB, ivec2(mod(position + dx.yx, RENDERSIZE)), 0);
        // vec4 d_l = texelFetch(bufferB, ivec2(mod(position - dx.yx, RENDERSIZE)), 0);

        // //position deltas
        // vec2 p_u = DECODE(d_u.x), p_d = DECODE(d_d.x);
        // vec2 p_r = DECODE(d_r.x), p_l = DECODE(d_l.x);

        // //velocities
        // vec2 v_u = DECODE(d_u.y), v_d = DECODE(d_d.y);
        // vec2 v_r = DECODE(d_r.y), v_l = DECODE(d_l.y);

        // //pressure gradient
        // vec2 p = vec2(Pressure(d_r) - Pressure(d_l),
        //                 Pressure(d_u) - Pressure(d_d));

        // //velocity operators
        // float div = (v_r.x - v_l.x + v_u.y - v_d.y);
        // float curl = (v_r.y - v_l.y - v_u.x + v_d.x);


       	gl_FragColor=sin(vec4(1,2,3,4)*1.2*r);
        gl_FragColor.a = 1.;
       	//col.xyz += vec3(1,0.1,0.1)*max(curl,0.) + vec3(0.1,0.1,1.)*max(-curl,0.);
    }
}
