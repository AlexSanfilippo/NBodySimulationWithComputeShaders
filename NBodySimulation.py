"""
Particle System using Compute Shaders and Instanced Rendering

-We have 1million particles working now.  this second file will be deconstructed to make a compute shader particle
 system class.
TODO

    ===steps below left for another script
    -[V] create new subclass
    -[V] create new shaders
    -[V] billboarding
    -[V] 4-vertex quads
    -[V]pass width to the 2 shaders as uniform
    -[V] initializing in different shapes, velocities
        -[V]Circle
        -[V]Disk (add random height range to circle)
        -[V]initialize velocity
        -[V]color by velocity?
    -[V] use open cv or other to save to video file
        -nb: we did this before in the sorting vectors project last autumn
    -[V] different mass per particle
        -store in velocity's alpha channel
        -effect particle scaling
#todo: Spring 2024
    -[]put this into a new pycharm project
    -[]upload to a public github repository
        -[]share to 3 requesters
    -[]readme
        -[]how to get this running
        -[]how to use
    -[]refactoring
        -[]change system size in ONE variable change
    -[]controls in one place (ie, a config file)
        -[]system size
        -[]particle color
        -[]spawn shape
            -[]size of spawn shape, etc.
        -[]gravity
        -[]particle mass
        -[]particle size



created: October 29, 2023
last update: March 7, 2024
"""

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from TextureLoader import load_texture
import numpy as np
import math
from camera import Camera, FollowCamera
import FPSCounter
import quad
import ComputeShaderParticleSystem

# cam = Camera(camera_pos=[0.0, 0.0, -10900.0], yaw=90.0, pitch=0.0)
cam = FollowCamera(camera_pos=[0.0, 0.0, -2900.0], yaw=90.0, pitch=0.0, target_position=[0.0, 0.0, 0.0])
WIDTH, HEIGHT = 800, 800 #1280, 720
# WIDTH, HEIGHT = 1280, 720
lastX, lastY = WIDTH / 2, HEIGHT / 2
first_mouse = True
left, right, forward, backward, pause = False, False, False, False, False


# the keyboard input callback
def key_input_clb(window, key, scancode, action, mode):
    global left, right, forward, backward, pitch_pos, draw_new_cube, pause, reset
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False
    if key == glfw.KEY_E and action == glfw.PRESS:
        pitch_pos = True
    elif key == glfw.KEY_E and action == glfw.RELEASE:
        pitch_pos = False
    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False
    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False
    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False
    if key == glfw.KEY_P and action == glfw.PRESS:
        pause = not pause
    if key == glfw.KEY_1 and action == glfw.PRESS:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    if key == glfw.KEY_2 and action == glfw.PRESS:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)


def mouse_button_callback(window, button, action, mods):
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        mpos = glfw.get_cursor_pos(window)
        print("Cursor Position at (", mpos[0], " : ", mpos[1], ")")


# do the movement, call this function in the main loop
def do_movement(speed=12.2):
    if left:
        cam.process_keyboard("LEFT", speed)
    if right:
        cam.process_keyboard("RIGHT", speed)
    if forward:
        cam.process_keyboard("FORWARD", speed)
    if backward:
        cam.process_keyboard("BACKWARD", speed)


def mouse_look_clb(window, xpos, ypos):
    global first_mouse, lastX, lastY
    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False
    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos
    cam.process_mouse_movement(xoffset, yoffset)


compute_shader_position_velocity = """
# version 440
/**
 Working Particle Mover 
*/

//input is one pixel of the image
//work group
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; //nb: must have a layout defined for shader to compile

//output to the image.  format must match here and in host program
layout(rgba32f, binding = 0) uniform image2D position; 
layout(rgba32f, binding = 1) uniform image2D velocity; 

uniform float time;
uniform vec3 acceleration;
uniform float max_lifespan;
uniform float draw_count;

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.

uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

// Compound versions of the hashing algorithm I whipped together.
uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }

void main()
{
    //absolute texel coord (ie, not normalized)
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    vec4 pos = imageLoad(position, texelCoord); //load pixel
    vec4 vel = imageLoad(velocity, texelCoord); //load pixel
    
    bool kill = false;
    
    //spawn into initial position + offset 100 on z-axis
    vec3 initial_position = vec3(gl_GlobalInvocationID.x*10.f, gl_GlobalInvocationID.y*10.f, 100.f);
    //respawn at world origin
    //vec3 initial_position = vec3(0.f, 0.f, 0.f);
    
    if(gl_GlobalInvocationID.x < draw_count*2 && gl_GlobalInvocationID.y < draw_count*2){    
        if(pos.a < 0.f){
            if(kill == false){
                //reset velocity and position
                imageStore(position, texelCoord, vec4(initial_position, max_lifespan));
                float speed = 10.f;
                vec3 random_velocity = speed*vec3(random(pos.x)/2. - 0.25, random(pos.y)/2. - 0.25, random(pos.z)/2. - 0.25);
                imageStore(velocity, texelCoord, vec4(random_velocity, 1.f));
            }
            else{
                //reset pos and vel
                imageStore(position, texelCoord,  vec4(initial_position, -1.f));
                imageStore(velocity, texelCoord, vec4(0.0, 0.0, 0.0, 0.0));
            }
        }
        else{         
            //add position to velocity to move particles
            vec4 next_position = vec4(pos.x + vel.x, pos.y + vel.y, pos.z + vel.z , pos.a - 1.f);  //correct, rm temp
            //vec4 next_position = vec4(pos.x , pos.y , pos.z , pos.a - 1.f);  //no vel
            imageStore(position, texelCoord, next_position);
            //add acceleration to velocity
            float dampening = 0.9983;
            imageStore(velocity, texelCoord, dampening*(vel + vec4(acceleration.xyz, 0.0)));
    
        }
    }
}
"""

compute_shader_initializer = """
# version 440
/**
 First attempt at any compute shader in PyOpenGL
*/

//input is one pixel of the image
//work group
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; //nb: must have a layout defined for shader to compile

//output to the image.  format must match here and in host program
layout(rgba32f, binding = 0) uniform image2D position; 
layout(rgba32f, binding = 1) uniform image2D velocity; 

uniform float time;
uniform float lifespan;

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.

uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

// Compound versions of the hashing algorithm I whipped together.
uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }

void main()
{
    //our new random method
    float position_spacing = 2000.f;
    vec2  inputsx = vec2( gl_GlobalInvocationID.xy); // Spatial and temporal inputs
    float randx = random( inputsx );              // Random per-pixel value
    vec2  inputsy = vec2( gl_GlobalInvocationID.yx); // Spatial and temporal inputs
    float randy = random( inputsy );              // Random per-pixel value
    vec2  inputsz = vec2( gl_GlobalInvocationID.yx*gl_GlobalInvocationID.x); // Spatial and temporal inputs
    float randz = random( inputsz );              // Random per-pixel value
    vec3  luma = vec3( randx*position_spacing, randy*position_spacing, randz*position_spacing ); // Expand to RGB

    float speed = 300.f;
    vec3 luma_velocity = speed*vec3(random(randx)/2.f - 0.25f, random(randy)/2.f - 0.25f, random(randz)/2.f - 0.25f);

    //absolute texel coord (ie, not normalized)
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);

    //write to image, at this texelCoord, the 4f vector of color data 
    
    //same starting point
    //vec4 same_position = vec4(0.f, 0.f, 0.f, lifespan);
    //imageStore(position, texelCoord, same_position);
    //id as position
    vec4 id_position = vec4((gl_GlobalInvocationID.x)*10, (gl_GlobalInvocationID.y)*10, 0.f, lifespan);
    imageStore(position, texelCoord, id_position);
    //New RNG function using uint hashing function
    //imageStore(position, texelCoord, vec4(luma, lifespan));
    
    imageStore(velocity, texelCoord, vec4(luma_velocity, 1.f));
}
"""




"""============================================="""

compute_shader_updater_class = """
# version 440
/**
 First attempt at any compute shader in PyOpenGL
*/

//input is one pixel of the image
//work group
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; //nb: must have a layout defined for shader to compile

//output to the image.  format must match here and in host program
layout(rgba32f, binding = 0) uniform image2D position; 
layout(rgba32f, binding = 1) uniform image2D velocity; 

uniform float time;

float old_random(vec2 st){
    return (fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123)/2.f) + 0.5f;
}

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.

uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

// Compound versions of the hashing algorithm I whipped together.
uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }

//system_size
#define PARTICLE_SYSTEM_SIZE 2000

void main()
{
    //absolute texel coord (ie, not normalized)
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    vec4 pos = imageLoad(position, texelCoord); //load pixel
    vec4 vel = imageLoad(velocity, texelCoord); //load pixel

    vec3 acceleration = vec3(0.f, 0.f, 0.f);
    vec3 sum_force = vec3(0.f);
    float min_distance_factor = 50.01;
    /*GRAVITY CALCULATION GOES HERE*/
    //for every other particle/workgroup   
    for(int n = 0; n < PARTICLE_SYSTEM_SIZE; n++){
        int i = int(float(n) / sqrt(float(PARTICLE_SYSTEM_SIZE)));
        int j = n % int(sqrt(float(PARTICLE_SYSTEM_SIZE)));
        //get the position of that work group
        vec4 pos_other = imageLoad(position, ivec2(i,j)); //load pixel

        //get the distance between this particle and the other one
        vec3 vector_from = pos_other.xyz - pos.xyz;

        //calculate the force due to gravity
        float denominator = pow(pow(length(vector_from),2.f) + min_distance_factor*min_distance_factor, 1.5); 
        //keep a running count of the net gravitational force
        sum_force += (pos_other.a * vector_from) /denominator;
    }
    
        
    //set acceleration according to the sum net force
    float gravitational_constant = 9.81;
    acceleration = gravitational_constant * sum_force; 

    //add position to velocity to move particles
    vec4 next_position = vec4(pos.xyz + vel.xyz , pos.a);  
    imageStore(position, texelCoord, next_position);
    //add acceleration to velocity
    float velocity_dampening = .987f;//.9887f;
    imageStore(velocity, texelCoord, velocity_dampening*vel + vec4(acceleration.xyz, 0.0));
}
"""

compute_shader_updater_class_disk = """
# version 440
/**
 First attempt at any compute shader in PyOpenGL
*/

//input is one pixel of the image
//work group
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; //nb: must have a layout defined for shader to compile

//output to the image.  format must match here and in host program
layout(rgba32f, binding = 0) uniform image2D position; 
layout(rgba32f, binding = 1) uniform image2D velocity; 

uniform float time;

float old_random(vec2 st){
    return (fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123)/2.f) + 0.5f;
}

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.

uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

// Compound versions of the hashing algorithm I whipped together.
uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }

void main()
{
    //absolute texel coord (ie, not normalized)
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    vec4 pos = imageLoad(position, texelCoord); //load pixel
    vec4 vel = imageLoad(velocity, texelCoord); //load pixel

    vec3 acceleration = vec3(0.f, 0.f, 0.f);
    vec3 sum_force = vec3(0.f);
    float min_distance_factor = 500.01;
    /*GRAVITY CALCULATION GOES HERE*/
    //for every other particle/workgroup

    for(int i = 0; i < 315; i++){
        for(int j = 0; j < 315; j++){

            //get the position of that work group
            vec4 pos_other = imageLoad(position, ivec2(i,j)); //load pixel

            //get the distance between this particle and the other one
            vec3 vector_from = pos_other.xyz - pos.xyz;

            //calculate the force due to gravity
            float denominator = pow(pow(length(vector_from),2.f) + min_distance_factor*min_distance_factor, 1.5); 
            sum_force += (pos_other.a * vector_from) /denominator;
            //keep a running count of the net gravitational force
        }
    }

    /*BIG mass in center of disk*/
    
    vec4 central_mass = vec4(vec3(0.f), 1000.f); //load pixel

    //get the distance between this particle and the other one
    vec3 vector_from = central_mass.xyz - pos.xyz;

    //calculate the force due to gravity
    float denominator = pow(pow(length(vector_from),2.f) + min_distance_factor*min_distance_factor, 1.5); 
    sum_force += (central_mass.a * vector_from) /denominator;
    

    //set acceleration according to the sum net force
    float gravitational_constant = 9.81;
    acceleration = gravitational_constant * sum_force; 

    //add position to velocity to move particles
    vec4 next_position = vec4(pos.x + vel.x, pos.y + vel.y, pos.z + vel.z , pos.a);  
    imageStore(position, texelCoord, next_position);
    //add acceleration to velocity
    float velocity_dampening = 0.9887f;//1.0or .9887f;
    imageStore(velocity, texelCoord, velocity_dampening*vel + vec4(acceleration.xyz, 0.0));
}
"""

compute_shader_initializer_class_disk = """
# version 440
/**
 Spawn particles in disk with correct velocity
*/

//input is one pixel of the image
//work group
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; //nb: must have a layout defined for shader to compile

//output to the image.  format must match here and in host program
layout(rgba32f, binding = 0) uniform image2D position; 
layout(rgba32f, binding = 1) uniform image2D velocity; 

uniform float time;

float random_old(vec2 st){
    return (fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123))*0.5 - 0.5;
}

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.

uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

// Compound versions of the hashing algorithm I whipped together.
uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }

#define M_PI 3.1415926535897932384626433832795

void main()
{
    //our new random method
    vec2  inputsx = vec2( gl_GlobalInvocationID.xy); // Spatial and temporal inputs
    float randx = random( inputsx );              // Random per-pixel value
    vec2  inputsy = vec2( gl_GlobalInvocationID.yx*0.34535345); // Spatial and temporal inputs
    float randy = random( inputsy );              // Random per-pixel value
    vec2  inputsz = vec2( gl_GlobalInvocationID.yx*gl_GlobalInvocationID.x); // Spatial and temporal inputs
    float randz = random( inputsz );              // Random per-pixel value
    float position_spacing = 2000.f;
    vec3 luma = vec3( randx*position_spacing, randy*position_spacing, randz*position_spacing ); // Expand to RGB

    float speed = 30.f;
    //vec3 luma_velocity = speed*vec3(random(randx)/2.f - 0.25f, random(randy)/2.f - 0.25f, random(randz)/2.f - 0.25f);
    vec3 luma_velocity = vec3(0.f, 0.f, 0.f);

    //absolute texel coord (ie, not normalized)
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);

    //write to image, at this texelCoord, the 4f vector of color data 

    //make this rng based later on
    float mass = 100.f;// * random(luma.x);

    //same starting point
    //vec4 same_position = vec4(0.f, 0.f, 0.f, mass);
    //imageStore(position, texelCoord, same_position);
    //id as position
    //vec4 id_position = vec4((gl_GlobalInvocationID.x)*10, (gl_GlobalInvocationID.y)*10, 0.f, mass);
    //imageStore(position, texelCoord, id_position);
    //New RNG function using uint hashing function
    //imageStore(position, texelCoord, vec4(luma, mass));
    
    //random point in sphere (working)
    /*
    vec3 point = vec3(randx-0.5f,randy-.5f,randz-.5f );
    float magnitude = length(point);
    point /= magnitude;
    float radius = 550.0f; //good planets at 250
    point *= radius;
    imageStore(position, texelCoord, vec4(point, mass));
    */
    
    /*Spawn in Disk*/
    /* usage in parameter of this class:
    spawn_shape=["CIRCLE", 800.0, [0.45, 0.2, 0.91], [0.0, 0.0, 0.0]],
    */
    float radius = 4050.f; //550
    float spin_speed = 0.005f;
    
    vec3 normal_vector = normalize(vec3(0.45f, 0.2f, 0.91f));
    //vec3 normal_vector = vec3(0.001f, 0.0001f, 1.f); //perpendicular to camera
    vec3 center = vec3(0.f, 0.f, 0.f);
    radius = pow(randx,4.0f) * radius; //more even distribution 
    vec3 point = vec3(0.f, 0.f, 0.f);

    //Graham-Schmidt: Generate 2 normalized & orthogonal vectors to the the normal
    //so with the normal, we have 3 mutually orthogonal vectors
    
    vec3 orthogonal_first = vec3(randx-.001f,randy-.001f,randz-.001f);
    orthogonal_first -= dot(orthogonal_first, normal_vector) * normal_vector;
    orthogonal_first = normalize(orthogonal_first);  // normalize it
    vec3 orthogonal_second = cross(normal_vector, orthogonal_first);
    //Now, use the two base vectors to get our point on the circle
    float rho = 2.f * M_PI * random(fract(randx*randy*randz));
    point.x = center.x + radius * orthogonal_first.x * cos(rho) + radius * orthogonal_second.x * sin(rho);
    point.y = center.y + radius * orthogonal_first.y * cos(rho) + radius * orthogonal_second.y * sin(rho);
    point.z = center.z + radius * orthogonal_first.z * cos(rho) + radius * orthogonal_second.z * sin(rho);

    /*NEW Velocity calculation*/
    float distance_to_center = distance(point, center);
    vec3 point_to_center = point - center;
    float delta = 1.f/length(point);
    float dummy = 10000;
    //vec3 disk_velocity = (0.05) * cross(normal_vector, point_to_center);
    float disk_velocity_magnitude = 0.05*(pow(1.f - distance_to_center/9000.f, 4.f));
    vec3 disk_velocity = disk_velocity_magnitude * cross(normal_vector, point_to_center);
    
    /*add noise*/
    point += random(point)*5.f;

    imageStore(velocity, texelCoord, vec4(0.f, 0.f, 0.f, 0.f));
    imageStore(velocity, texelCoord, vec4(disk_velocity, 0.f));
    imageStore(position, texelCoord, vec4(point, mass));
    
}
"""
compute_shader_initializer_class = """
# version 440
/**
 First attempt at any compute shader in PyOpenGL
*/

//input is one pixel of the image
//work group
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; //nb: must have a layout defined for shader to compile

//output to the image.  format must match here and in host program
layout(rgba32f, binding = 0) uniform image2D position; 
layout(rgba32f, binding = 1) uniform image2D velocity; 

uniform float time;

float random_old(vec2 st){
    return (fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123))*0.5 - 0.5;
}

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.

uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

// Compound versions of the hashing algorithm I whipped together.
uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }

void main()
{
    //our new random method
    vec2  inputsx = vec2( gl_GlobalInvocationID.xy); // Spatial and temporal inputs
    float randx = random( inputsx );              // Random per-pixel value
    vec2  inputsy = vec2( gl_GlobalInvocationID.yx*0.34535345); // Spatial and temporal inputs
    float randy = random( inputsy );              // Random per-pixel value
    vec2  inputsz = vec2( gl_GlobalInvocationID.yx*gl_GlobalInvocationID.x); // Spatial and temporal inputs
    float randz = random( inputsz );              // Random per-pixel value
    float position_spacing = 2000.f;
    vec3 luma = vec3( randx*position_spacing, randy*position_spacing, randz*position_spacing ); // Expand to RGB

    float speed = 30.f;
    vec3 luma_velocity = speed*vec3(random(randx)/2.f - 0.25f, random(randy)/2.f - 0.25f, random(randz)/2.f - 0.25f);
    //vec3 luma_velocity = vec3(0.f, 0.f, 0.f);

    //absolute texel coord (ie, not normalized)
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);

    //write to image, at this texelCoord, the 4f vector of color data 

    float mass = 10.f * random(luma.x);

    //same starting point
    //vec4 same_position = vec4(0.f, 0.f, 0.f, mass);
    //imageStore(position, texelCoord, same_position);
    //id as position
    //vec4 id_position = vec4((gl_GlobalInvocationID.x)*50, (gl_GlobalInvocationID.y)*50, 0.f, mass);
    //imageStore(position, texelCoord, id_position);
    //New RNG function using uint hashing function
    //imageStore(position, texelCoord, vec4(luma, mass));

    //random point in sphere (working)

    vec3 point = vec3(randx-0.5f,randy-.5f,randz-.5f );
    //vec3 point = vec3(randx-0.5,randy-0.5f,randz );  //half sphere
    float magnitude = length(point);
    point /= magnitude;
    float radius = 1050.0f; //good planets at 250
    point *= radius;
    imageStore(position, texelCoord, vec4(point, mass));


    //imageStore(velocity, texelCoord, vec4(luma_velocity, 1.f));
    imageStore(velocity, texelCoord, vec4(0.f, 0.f, 0.f, 0.f));


}
"""

vertex_shader_n_body = """
# version 440
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_text_coord;
layout(binding=0) uniform sampler2D position;
layout(binding=1) uniform sampler2D velocity;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

out vec4 compute_shader_color;

void main()
{
    //go from 1-dim gl_InstanceID to 2D texture coordinates
    float width = 50.001f; //must be ever so slightly larger than texture to index properly.  WHY?!
    float id = gl_InstanceID;

    //NORMALIZED TEXTURE COORDINATES 
    vec2 position_texture_coords = vec2(float(mod(id, width))/width, float(floor(id / width))/width);
    vec4 particle_position = texture(position, position_texture_coords);

    //compute_shader_color = vec4(position_texture_coords.x, position_texture_coords.y, 1.0, 1.0);
    compute_shader_color = vec4(0.05f, 0.2f, 0.2f, 1.0);

    gl_Position = projection * view * model * vec4(a_position.x + particle_position.x, a_position.y + particle_position.y, a_position.z + particle_position.z, 1.0f);
}
"""

vertex_shader_n_body_billboard = """
# version 430
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_text_coord;
layout(binding=0) uniform sampler2D position;
layout(binding=1) uniform sampler2D velocity;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

out vec4 compute_shader_color;

void main()
{
    //go from 1-dim gl_InstanceID to 2D texture coordinates
    //update_particle_texture_size
    //system_size
    float width = 315.001f; //must be ever so slightly larger than texture to index properly.  WHY?!
    float id = gl_InstanceID;

    //NORMALIZED TEXTURE COORDINATES 
    vec2 position_texture_coords = vec2(float(mod(id, width))/width, float(floor(id / width))/width);
    vec4 particle_position = texture(position, position_texture_coords);
    vec4 particle_velocity = texture(velocity, position_texture_coords);

    //compute_shader_color = vec4(0.1f, 0.4f, 0.6, 1.0f);  //nice blue/cyan color
    compute_shader_color = vec4(0.922f, 0.31f, 0.141f, 1.0f);  //nice red/orange color (praise the sun)
    //compute_shader_color = vec4(particle_velocity.xyz, 1.f); //color by velocity
    //compute_shader_color = vec4(position_texture_coords.x,position_texture_coords.y, 1.0f, 1.0);//color by work group

    mat4 bb_model = mat4(1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0, 
                         particle_position.x, particle_position.y, particle_position.z, 1.0);
    mat4 mvmat = view * bb_model;
    mvmat[0][0] = mvmat[1][1] = mvmat[2][2] = 1.0f;
    mvmat[0][1] = mvmat[0][2] = mvmat[1][2] = 0.0f;
    mvmat[1][0] = mvmat[2][0] = mvmat[2][1] = 0.0f;

    float scale = 1.f; 
    gl_Position = projection * mvmat * vec4(a_position*scale, 1.0f);
}
"""

fragment_shader_n_body = """
# version 440

in vec4 compute_shader_color;
out vec4 out_color;

void main()
{
    out_color = compute_shader_color;
}
"""


# the window resize callback function
def window_resize_clb(window, width, height):
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 1000)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)


# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")


# creating the window
window = glfw.create_window(WIDTH, HEIGHT, "N-Body Gravity Simulation", None, None)


# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")


# set window's position
glfw.set_window_pos(window, 400, 100)
# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize_clb)
# set the mouse position callback
glfw.set_cursor_pos_callback(window, mouse_look_clb)
# set the keyboard input callback
glfw.set_key_callback(window, key_input_clb)
# capture mouse button input
glfw.set_mouse_button_callback(window, mouse_button_callback)
# capture the mouse cursor
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

"""use OpenGL 4.3"""
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 4)


# make the context current
glfw.make_context_current(window)


# glClearColor(0.2, 0.3, 0.3, 1) # "dev's teal"
glClearColor(0.1, 0.1, 0.1, 1) # "show-off's black"
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)


my_fps = FPSCounter.FPSCounter(frame_interval=300.0)


"""New Compute Particle Shader class"""
dimensions = [2.0]*2
quad_vertices = np.array([
     1.0*dimensions[0],  1.0*dimensions[1], 0.0, 1.0, 1.0,
    -1.0*dimensions[0],  1.0*dimensions[1], 0.0, 0.0, 1.0,
    -1.0*dimensions[0], -1.0*dimensions[1], 0.0, 0.0, 0.0,
     1.0*dimensions[0], -1.0*dimensions[1], 0.0, 1.0, 0.0,],
    dtype=np.float32
)
quad_indices = np.array([
    0, 3, 2,
    0, 1, 2,],
    dtype=np.uint32
)


particle_system = ComputeShaderParticleSystem.NBodyParticleSystem(
    vertex_shader=vertex_shader_n_body_billboard,
    fragment_shader=fragment_shader_n_body,
    compute_shader_initializer=compute_shader_initializer_class,
    compute_shader_updater=compute_shader_updater_class,
    vertices=quad_vertices,
    indices=quad_indices,
    projection=pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, 200000),
    texture_dimensions=[315, 315],#[315, 315], #system_size
    draw_mode=GL_TRIANGLES,
)

"""For drawing GL_POINTS, can set the size as such"""
glPointSize(1.0)


"""experimental gif writing"""
from PIL import Image, ImageDraw
images = []


write_to_gif = False
frame_count = 0
# while not glfw.window_should_close(window) and frame_count < 1786:
# while not glfw.window_should_close(window) and frame_count < 50:
while not glfw.window_should_close(window):
    frame_count += 1
    """General OpenGL Stuff"""
    glfw.poll_events()
    do_movement(40.0 * 60.0 * my_fps.update())
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)



    if not pause:
        particle_system.update()

    view = cam.get_view_matrix()
    """NEW rotation camera"""
    cam.rotate_camera_over_time(speed = .25)
    particle_system.draw(view)

    """Write window to gif using PIL"""
    if write_to_gif:
        window_as_numpy_arr = np.uint8(glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_FLOAT) * 255.0)
        window_as_PIL_image = Image.fromarray(window_as_numpy_arr)
        images.append(window_as_PIL_image)
    
    glfw.swap_buffers(window)

if write_to_gif:
    images[0].save(
            'SuperNova100k_rotation_a.gif',
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=35,
            loop=0
        )
print("FINAL FRAME COUNT: ", frame_count)
glfw.terminate()
