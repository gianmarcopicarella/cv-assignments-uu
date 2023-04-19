import os
import glm
import json
import numpy as np
from OpenGL.GL import *
from engine.renderable.mesh import Mesh


class Model:
    def __init__(self, path, rotation_mat=glm.mat4(1)):
        self.meshes = []
        if not os.path.exists(path):
            raise RuntimeError(f'Model source file {path} does not exists.')
        self.path = path

        if path in "cube":
            self.model = glm.mat4()
            #self.model[0][0] = 0.5
            #self.model[1][1] = 0.5
            #self.model[2][2] = 0.5
        else:
            self.model = glm.mat4()


        self.rotation = rotation_mat
        data = self._load_get_data()
        for meshData in data['meshes']:
            self.meshes.append(Mesh(meshData))

    def _load_get_data(self):
        with open(self.path) as file:
            data = json.load(file)
        return data

    def set_multiple_positions(self, positions, colors):
        for mesh in self.meshes:
            mesh.set_multiple_positions(positions, colors)

    def draw(self, program):
        program.use()
        program.setMat4('model', self.model)
        for mesh in self.meshes:
            mesh.draw()

    def draw_multiple(self, program):
        program.use()
        program.setMat4('model', self.model)
        program.setMat4('rotation', self.rotation)
        for mesh in self.meshes:
            mesh.draw_multiple()

    def __del__(self):
        self.delete()

    def delete(self):
        self.meshes.clear()
