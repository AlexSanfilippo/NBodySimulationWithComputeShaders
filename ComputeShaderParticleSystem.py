"""
A class for a particle sytem that runs on compute shaders

Started 12th Nov, 2023
Alexander Sanfilippo
TODO:
-[] set up the render pipeline (get one particle system running)
-[]Two simulaneous instances of this class running
-[] N-body subclass
-[] Shaped-spawn subclass

"""
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from TextureLoader import load_texture
import numpy as np
import math


class ComputeShaderParticleSystem:
    """
    Creates a particle system which runs on compute shaders.  How parallel!
    """

    def __init__(
        self,
        vertex_shader,
        fragment_shader,
        compute_shader_initializer,
        compute_shader_updater,
        vertices,
        indices,
        projection,
        position = [0.0, 0.0, 0.0],
        acceleration = [0.09, 0.0, -0.09],
        texture_dimensions = [512, 512],
        draw_mode = GL_TRIANGLES,
        lifespan = 300,
        window_dimensions=[800,400],
        system_id=0
    ):
        self.position = np.array(position)
        self.acceleration = np.array(acceleration)
        self.texture_dimensions = texture_dimensions
        self.draw_mode = draw_mode
        self.lifespan = lifespan
        self.vertices = vertices
        self.indices = indices
        self.window_dimensions = window_dimensions
        self.projection = projection
        self.draw_count = 0
        self.system_id = system_id

        """Create Shaders"""
        self.shader = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )
        self.compute_shader_updater = compileProgram(compileShader(compute_shader_updater, GL_COMPUTE_SHADER))
        self.compute_shader_initializer = compileProgram(compileShader(compute_shader_initializer, GL_COMPUTE_SHADER))

        """set up posiiton and velocity textures"""
        self.create_textures()

        """set up buffers and shader attributes"""
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        # POSITION ATTRIBUTE
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 5, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # TEXTURE COORDINATE ATTRIBUTE
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 5, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)

        """Projection matrix pass to shader"""
        glUseProgram(self.shader)
        proj_loc = glGetUniformLocation(self.shader, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, self.projection)

        """run initializing compute shader"""
        glUseProgram(self.compute_shader_initializer)
        time_loc = glGetUniformLocation(self.compute_shader_initializer, "time")
        glUniform1f(time_loc, glfw.get_time())
        lifespan_loc = glGetUniformLocation(self.compute_shader_initializer, "lifespan")
        glUniform1f(lifespan_loc, self.lifespan)
        glDispatchCompute(self.texture_dimensions[0], self.texture_dimensions[1], 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)




    def draw(self, view):
        """
        Calls OpenGL Draw functions and setup junk
        :param view: view matrix from the camera
        :return: no return
        """
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # additive blending

        # print(view)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, view)

        self.bind_textures()


        glDrawElementsInstanced(
            self.draw_mode,
            len(self.indices),
            GL_UNSIGNED_INT,
            None,
            self.texture_dimensions[0] * self.texture_dimensions[1],
        )

    def bind_textures(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_0_position)
        glBindImageTexture(0, self.texture_0_position, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.texture_1_velocity)
        glBindImageTexture(1, self.texture_1_velocity, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)


    def update(self):
        """Use Compute Shader"""
        if self.draw_count < self.texture_dimensions[0]:
            self.draw_count += 1
        glUseProgram(self.compute_shader_updater)


        """texture binding"""
        self.bind_textures()

        time_loc = glGetUniformLocation(self.compute_shader_updater, "time")
        glUniform1f(time_loc, glfw.get_time())
        acceleration_loc = glGetUniformLocation(self.compute_shader_updater, "acceleration")  # TODO remove this here & shader
        glUniform3fv(acceleration_loc, 1, self.acceleration)
        max_lifespan_loc = glGetUniformLocation(self.compute_shader_updater, "max_lifespan")
        glUniform1f(max_lifespan_loc, self.lifespan)
        draw_count_loc = glGetUniformLocation(self.compute_shader_updater, "draw_count")
        glUniform1f(draw_count_loc, self.draw_count)
        glDispatchCompute(self.texture_dimensions[0], self.texture_dimensions[1], 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    def create_textures(self):
        self.texture_0_position = glGenTextures(1)
        """Texture that Compute and Vertex Shaders share"""
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_0_position)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        initial_texture_values_red = np.array([1] * self.texture_dimensions[0] * self.texture_dimensions[1], dtype=np.float32)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.texture_dimensions[0], self.texture_dimensions[1], 0, GL_RED, GL_FLOAT,
                     initial_texture_values_red)
        glBindImageTexture(0, self.texture_0_position, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)

        self.texture_1_velocity = glGenTextures(1)
        """Texture that Compute and Vertex Shaders share"""
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.texture_1_velocity)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        initial_texture_values_red = np.array([1] * self.texture_dimensions[0] * self.texture_dimensions[1], dtype=np.float32)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.texture_dimensions[0], self.texture_dimensions[1], 0, GL_RED, GL_FLOAT,
                     initial_texture_values_red)
        glBindImageTexture(1, self.texture_1_velocity, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)


class NBodyParticleSystem(ComputeShaderParticleSystem):
    """
    Compute-Shader-Driven 3D N-Body Gravity simulation
    """

    def __init__(
        self,
        vertex_shader,
        fragment_shader,
        compute_shader_initializer,
        compute_shader_updater,
        vertices,
        indices,
        projection,
        position = [0.0, 0.0, 0.0],
        acceleration = [0.09, 0.0, -0.09],
        texture_dimensions = [512, 512],
        draw_mode = GL_TRIANGLES,
        lifespan = 300,
        window_dimensions=[800,400],
        system_id=0
    ):
        self.position = np.array(position)
        self.acceleration = np.array(acceleration)
        self.texture_dimensions = texture_dimensions
        self.draw_mode = draw_mode
        self.lifespan = lifespan
        self.vertices = vertices
        self.indices = indices
        self.window_dimensions = window_dimensions
        self.projection = projection
        self.draw_count = texture_dimensions[0]*texture_dimensions[1]
        self.system_id = system_id

        """Create Shaders"""
        self.shader = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )
        self.compute_shader_updater = compileProgram(compileShader(compute_shader_updater, GL_COMPUTE_SHADER))
        self.compute_shader_initializer = compileProgram(compileShader(compute_shader_initializer, GL_COMPUTE_SHADER))

        """set up posiiton and velocity textures"""
        self.create_textures()

        """set up buffers and shader attributes"""
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        # POSITION ATTRIBUTE
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 5, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # TEXTURE COORDINATE ATTRIBUTE
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 5, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)

        """Projection matrix pass to shader"""
        glUseProgram(self.shader)
        proj_loc = glGetUniformLocation(self.shader, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, self.projection)
        self.view_loc = glGetUniformLocation(self.shader, "view")


        """run initializing compute shader"""
        glUseProgram(self.compute_shader_initializer)
        time_loc = glGetUniformLocation(self.compute_shader_initializer, "time")
        glUniform1f(time_loc, glfw.get_time())
        lifespan_loc = glGetUniformLocation(self.compute_shader_initializer, "lifespan")
        glUniform1f(lifespan_loc, self.lifespan)
        glDispatchCompute(self.texture_dimensions[0], self.texture_dimensions[1], 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


    def update(self):
        """Use Compute Shader"""
        glUseProgram(self.compute_shader_updater)

        """texture binding"""
        self.bind_textures()

        time_loc = glGetUniformLocation(self.compute_shader_updater, "time")
        glUniform1f(time_loc, glfw.get_time())
        glDispatchCompute(self.texture_dimensions[0], self.texture_dimensions[1], 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

