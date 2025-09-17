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
        }
    ],
    "ISFVSN": "2",
    "PASSES": [
        {
            "TARGET": "bufferA",
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

// uint floatBitsToUint(float f);
// float uintBitsToFloat(uint f);
#define iFrame FRAMEINDEX
#define iResolution RENDERSIZE
#define U gl_FragColor
#define fragColor gl_FragColor
#define col gl_FragColor

// Constants and functions from LYGIA <https://github.com/patriciogonzalezvivo/lygia>
#define PI 3.1415926535897932384626433832795


//
// ShaderToy Common
//

#define T_A(p) texelFetch(bufferB, ivec2(mod(p,R)), 0)
#define T_B(p) texelFetch(bufferA, ivec2(mod(p,R)), 0)
#define P(p) texture(bufferB, mod(p,R)/R)
#define C_B(p) texture(bufferC, mod(p,R)/R)
#define C_D(p) texture(bufferD, mod(p,R)/R)
#define D(p) texture(iChannel2, mod(p,R)/R)

#define dt 1.
#define R iResolution.xy

const vec2 dx = vec2(0, 1);

float hash11(float p)
{
    p = fract(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

#define rand_interval 250
#define random_gen(a, b, seed) ((a) + ((b)-(a))*hash11(seed + float(iFrame/rand_interval)))

//i.e. diffusion
#define distribution_size 1.7

//slime mold sensors
#define sense_num 12
#define sense_ang 0.03
#define sense_dis 15.
//weird oscillation force
#define sense_oscil 0.0
#define oscil_scale 1.
#define oscil_pow 0.
#define sense_force 0.38
//slime mold sensor distance power law dist = sense_dispow(rho, pw)
#define distance_scale 0.0
#define force_scale 1.
#define acceleration 0.

#define density_normalization_speed 0.13
#define density_target 0.24
#define vorticity_confinement 0.0

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
#define fluid_rho 0.2


//data packing
#define PACK(X) ( uint(round(65534.0*clamp(0.5*X.x+0.5, 0., 1.))) + \
           65535u*uint(round(65534.0*clamp(0.5*X.y+0.5, 0., 1.))) )

#define UNPACK(X) (clamp(vec2(X%65535u, X/65535u)/65534.0, 0.,1.)*2.0 - 1.0)

#define DECODE(X) UNPACK(floatBitsToUint(X))
#define ENCODE(X) uintBitsToFloat(PACK(X))


void main()
{
    vec2 pos = gl_FragCoord.xy;

    if (PASSINDEX == 0) // ShaderToy Buffer A
    {
        ivec2 p = ivec2(pos);

        vec2 X = vec2(0);
        vec2 V = vec2(0);
        float M = 0.;

        //basically integral over all updated neighbor distributions
        //that fall inside of this pixel
        //this makes the tracking conservative
        range(i, -2, 2) range(j, -2, 2)
        {
            vec2 tpos = pos + vec2(i,j);
            vec4 data = T_A(tpos);

            vec2 X0 = DECODE(data.x) + tpos;
           	vec2 V0 = DECODE(data.y);
           	vec2 M0 = data.zw;

            X0 += V0*dt; //integrate position

            //particle distribution size
            float K = distribution_size;

            vec4 aabbX = vec4(max(pos - 0.5, X0 - K*0.5), min(pos + 0.5, X0 + K*0.5)); //overlap aabb
            vec2 center = 0.5*(aabbX.xy + aabbX.zw); //center of mass
            vec2 size = max(aabbX.zw - aabbX.xy, 0.); //only positive

            //the deposited mass into this cell
            float m = M0.x*size.x*size.y/(K*K);

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

        //initial condition
        if(iFrame < 1 || restart)
        {
            X = pos;
            vec2 dx0 = (pos - R*0.3); vec2 dx1 = (pos - R*0.7);
            V = 0.5*Rot(PI*0.5)*dx0*GS(dx0/30.) - 0.5*Rot(PI*0.5)*dx1*GS(dx1/30.);
            V += 1.0*Dir(2.*PI*hash11(floor(pos.x/10.) + R.x*floor(pos.y/20.)));
            M = 0.1 + pos.x/R.x*0.01 + pos.y/R.x*0.01;
        }

        X = clamp(X - pos, vec2(-0.5), vec2(0.5));
        U = vec4(ENCODE(X), ENCODE(V), M, 0.);
    }
    else if (PASSINDEX == 1) // ShaderToy Buffer B
    {
        vec2 uv = pos/R;
        ivec2 p = ivec2(pos);

        vec4 data = T_B(pos);
        vec2 X = DECODE(data.x) + pos;
        vec2 V = DECODE(data.y);
        float M = data.z;

        if(M != 0.) //not vacuum
        {
            //Compute the force
            vec2 F = vec2(0.);

            //get neighbor data
            vec4 d_u = T_B(pos + dx.xy), d_d = T_B(pos - dx.xy);
            vec4 d_r = T_B(pos + dx.yx), d_l = T_B(pos - dx.yx);

            //position deltas
            vec2 p_u = DECODE(d_u.x), p_d = DECODE(d_d.x);
            vec2 p_r = DECODE(d_r.x), p_l = DECODE(d_l.x);

            //velocities
            vec2 v_u = DECODE(d_u.y), v_d = DECODE(d_d.y);
            vec2 v_r = DECODE(d_r.y), v_l = DECODE(d_l.y);



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
            float dang =sense_ang*PI/float(sense_num);
            vec2 slimeF = vec2(0.);
            //slime mold sensors
            range(i, -sense_num, sense_num)
            {
                float cang = ang + float(i) * dang;
            	vec2 dir = (1. + sense_dis*pow(M, distance_scale))*Dir(cang);
            	vec3 s0 = C_B(X + dir).xyz;
       			float fs = pow(s0.z, force_scale);
                float os = oscil_scale*pow(s0.z - M, oscil_pow);
            	slimeF +=  sense_oscil*Rot(os)*s0.xy
                         + sense_force*Dir(ang + sign(float(i))*PI*0.5)*fs;
            }

            //remove acceleration component from slime force and leave rotation only
            slimeF -= dot(slimeF, normalize(V))*normalize(V);
    		F += slimeF/float(2*sense_num);

            // TODO
            // if(iMouse.z > 0.)
            // {
            //     vec2 dx= pos - iMouse.xy;
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
        //\\	M = mix(M, 0.5, GS((pos - iMouse.xy)/13.));
        //else
         //   M = mix(M, 0.5, GS((pos - R*0.5)/13.));

        //save
        X = clamp(X - pos, vec2(-0.5), vec2(0.5));
        U = vec4(ENCODE(X), ENCODE(V), M, 0.);
    }
    else if (PASSINDEX == 2) // ShaderToy Buffer C
    {
        float rho = 0.001;
        vec2 vel = vec2(0., 0.);

        //compute the smoothed density and velocity
        range(i, -2, 2) range(j, -2, 2)
        {
            vec2 tpos = pos + vec2(i,j);
            vec4 data = T_A(tpos);

            vec2 X0 = DECODE(data.x) + tpos;
            vec2 V0 = DECODE(data.y);
            float M0 = data.z;
            vec2 dx = X0 - pos;

#define radius 2.
            float K = GS(dx/radius)/(radius);
            rho += M0*K;
            vel += M0*K*V0;
        }

        vel /= rho;

        fragColor = vec4(vel, rho, 1.0);
    }
    else if (PASSINDEX == 3) // ShaderToy Buffer D
    {
        vec2 V0 = vec2(0.);
        if(iFrame%1 == 0)
        {
        	vec4 data = T_A(pos);
        	V0 = 1.*DECODE(data.y);
       		float M0 = data.z;
        }
        else
        {

        }

        fragColor = C_D(pos - V0*dt);
        //initial condition
        if(iFrame < 1 || restart)
        {
            fragColor.xy = pos/R;
        }
    }
    else // ShaderToy Image
    {
        float r = P(pos.xy).z;
       	vec4 c = C_D(pos.xy);

       	//get neighbor data
        vec4 d_u = T_A(pos + dx.xy), d_d = T_A(pos - dx.xy);
        vec4 d_r = T_A(pos + dx.yx), d_l = T_A(pos - dx.yx);

        //position deltas
        vec2 p_u = DECODE(d_u.x), p_d = DECODE(d_d.x);
        vec2 p_r = DECODE(d_r.x), p_l = DECODE(d_l.x);

        //velocities
        vec2 v_u = DECODE(d_u.y), v_d = DECODE(d_d.y);
        vec2 v_r = DECODE(d_r.y), v_l = DECODE(d_l.y);

        //pressure gradient
        vec2 p = vec2(Pressure(d_r) - Pressure(d_l),
                        Pressure(d_u) - Pressure(d_d));

        //velocity operators
        float div = (v_r.x - v_l.x + v_u.y - v_d.y);
        float curl = (v_r.y - v_l.y - v_u.x + v_d.x);


       	col=sin(vec4(1,2,3,4)*1.2*r);
       	//col.xyz += vec3(1,0.1,0.1)*max(curl,0.) + vec3(0.1,0.1,1.)*max(-curl,0.);
    }
}
