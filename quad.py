"""
    13th August, 2023
    Brief: A class hierarchy for creating and drawing a simple quad
"""

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
import numpy as np


class Quad:
    def __init__(self,
             shader,
             #material properties
             diffuse,
             specular,
             shininess = 32.0,
             #mesh properties
             position=[0.0, 0.0, 0.0],
             dimensions = [5.0, 5.0],
             ):
        self.shader = shader
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.position = position
        self.dimensions = dimensions

        self.vertices = np.array([
             1.0*dimensions[0],  1.0*dimensions[1], 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            -1.0*dimensions[0],  1.0*dimensions[1], 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            -1.0*dimensions[0], -1.0*dimensions[1], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
             1.0*dimensions[0],  1.0*dimensions[1], 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            -1.0*dimensions[0], -1.0*dimensions[1], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
             1.0*dimensions[0], -1.0*dimensions[1], 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32)

        self.indices = np.array([
            3, 4, 5,
            0, 1, 2,],
            dtype=np.uint32)

        # quad VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # Element Buffer Object
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        # quad position vertices (vertex attribute)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # quad texture coords
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        # quad normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)

    def draw(self, view, model_loc):
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.diffuse)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.specular)

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position)))
        glDrawArrays(GL_TRIANGLES, 0, len(self.indices))

    def set_diffuse(self, diffuse):
        self.diffuse = diffuse

    def set_specular(self, specular):
        self.specular = specular

    def get_diffuse(self):
        return self.diffuse

    def get_specular(self):
        return self.specular

    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position
        self.translation = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))

class QuadPositional():
    """
    Quad with only position attribute
    """
    def __init__(self,
             shader,
             position=[0.0, 0.0, 0.0],
             dimensions = [5.0, 5.0],
             window_dimensions = [800, 400]
             ):
        self.shader = shader
        self.position = position
        self.dimensions = dimensions
        self.window_dimensions = window_dimensions

        self.vertices = np.array([
             1.0*dimensions[0],  1.0*dimensions[1], 0.0,
            -1.0*dimensions[0],  1.0*dimensions[1], 0.0,
            -1.0*dimensions[0], -1.0*dimensions[1], 0.0,
             1.0*dimensions[0],  1.0*dimensions[1], 0.0,
            -1.0*dimensions[0], -1.0*dimensions[1], 0.0,
             1.0*dimensions[0], -1.0*dimensions[1], 0.0,],
            dtype=np.float32)

        self.indices = np.array([
            3, 4, 5,
            0, 1, 2,],
            dtype=np.uint32)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        # quad position vertices (vertex attribute)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        projection = pyrr.matrix44.create_perspective_projection_matrix(
            45,
            self.window_dimensions[0] / self.window_dimensions[1],
            0.1,
            20000
        )
        glUseProgram(self.shader)
        proj_loc = glGetUniformLocation(self.shader, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    def draw(self, view):
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        model_loc = glGetUniformLocation(self.shader, "model")
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position)))

        view_loc = glGetUniformLocation(self.shader, "view")
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

        glDrawArrays(GL_TRIANGLES, 0, len(self.indices))

    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position
        self.translation = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))

class QuadTextured(QuadPositional):
    """
    Quad with position and texture coordinate attributes
    """
    def __init__(self,
             shader,
             position=[0.0, 0.0, 0.0],
             dimensions = [5.0, 5.0],
             window_dimensions = [800, 400]
             ):
        self.shader = shader
        self.position = position
        self.dimensions = dimensions
        self.window_dimensions = window_dimensions

        self.vertices = np.array([
             1.0*dimensions[0],  1.0*dimensions[1], 0.0, 1.0, 1.0,
            -1.0*dimensions[0],  1.0*dimensions[1], 0.0, 0.0, 1.0,
            -1.0*dimensions[0], -1.0*dimensions[1], 0.0, 0.0, 0.0,
             1.0*dimensions[0],  1.0*dimensions[1], 0.0, 1.0, 1.0,
            -1.0*dimensions[0], -1.0*dimensions[1], 0.0, 0.0, 0.0,
             1.0*dimensions[0], -1.0*dimensions[1], 0.0, 1.0, 0.0,],
            dtype=np.float32)

        self.indices = np.array([
            3, 4, 5,
            0, 1, 2,],
            dtype=np.uint32
        )

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

        projection = pyrr.matrix44.create_perspective_projection_matrix(
            45,
            self.window_dimensions[0] / self.window_dimensions[1],
            0.1,
            20000
        )
        glUseProgram(self.shader)
        proj_loc = glGetUniformLocation(self.shader, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
