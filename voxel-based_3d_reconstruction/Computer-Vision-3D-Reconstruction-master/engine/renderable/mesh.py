import numpy as np
from OpenGL.GL import *
from OpenGL.error import NullFunctionError


class Mesh:
    def __init__(self, data):
        indices_list = self._get_indices_list(data['faces'])
        self.__indicesLen = len(indices_list)
        indices_data = np.array(indices_list, dtype=np.uint32)
        vertex_data = np.array(data['vertices'], dtype=np.float32)
        normal_data = np.array(data['normals'], dtype=np.float32)
        tex_coords_data = np.array(data['texturecoords'], dtype=np.float32)
        tangent_data = np.array(data['tangents'], dtype=np.float32)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_data, GL_STATIC_DRAW)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, vertex_data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        self.VBO_N = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO_N)
        glBufferData(GL_ARRAY_BUFFER, normal_data, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        self.VBO_TEX = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO_TEX)
        glBufferData(GL_ARRAY_BUFFER, tex_coords_data, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)

        self.VBO_TAN = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO_TAN)
        glBufferData(GL_ARRAY_BUFFER, tangent_data, GL_STATIC_DRAW)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(3)

        self.VBO_POS = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO_POS)
        data = np.identity(4, dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(4)
        self.positionsLen = 1

        self.VBO_COL = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO_COL)
        data = np.identity(4, dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(5)

        glVertexAttribDivisor(0, 0)
        glVertexAttribDivisor(1, 0)
        glVertexAttribDivisor(2, 0)
        glVertexAttribDivisor(3, 0)
        glVertexAttribDivisor(4, 1)
        glVertexAttribDivisor(5, 1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    @staticmethod
    def _get_indices_list(assimp_indices):
        indices_list = []
        for face in assimp_indices:
            for index in face:
                indices_list.append(index)
        return indices_list

    def set_multiple_positions(self, positions, colors):
        assert len(positions) == len(colors), f'len(positions), {len(positions)}, must be equal to len(colors), {len(colors)}'
        data = np.array(positions, dtype=np.float32)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO_POS)
        glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
        glBindVertexArray(0)

        data = np.array(colors, dtype=np.float32)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO_COL)
        glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
        glBindVertexArray(0)

        self.positionsLen = len(positions)

    def draw(self):
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, self.__indicesLen, GL_UNSIGNED_INT, None)

    def draw_multiple(self):
        glBindVertexArray(self.VAO)
        glDrawElementsInstanced(GL_TRIANGLES, self.__indicesLen, GL_UNSIGNED_INT, None, self.positionsLen)

    def __del__(self):
        self.delete()

    def delete(self):
        try:
            glDeleteVertexArrays(1, self.VAO)
            glDeleteBuffers(1, self.VBO)
            glDeleteBuffers(1, self.VBO_N)
            glDeleteBuffers(1, self.VBO_TEX)
            glDeleteBuffers(1, self.VBO_TAN)
            glDeleteBuffers(1, self.EBO)
            glDeleteBuffers(1, self.VBO_POS)
            glDeleteBuffers(1, self.VBO_COL)
            self.VAO, self.VBO, self.VBO_N, self.VBO_TEX, self.VBO_TAN, self.EBO, self.VBO_POS, self.VBO_COL = 0, 0, 0, 0, 0, 0, 0, 0
        except (NullFunctionError, TypeError):
            pass 
