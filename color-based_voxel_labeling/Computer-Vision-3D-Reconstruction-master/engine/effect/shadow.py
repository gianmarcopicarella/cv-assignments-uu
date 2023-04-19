import glm
from OpenGL.GL import *
from engine.buffer.depthbuffer import DepthBuffer
from engine.buffer.framebuffer import FrameBuffer


class Shadow:
    def __init__(self, light_pos, near_plane, far_plane):
        self.frame_buffer = None
        self.depth_buffer = None
        self.height = None
        self.width = None
        self.lightSpaceMatrix = None
        self.update_matrix(light_pos, near_plane, far_plane)

    def update_matrix(self, light_pos, near_plane, far_plane):
        light_projection = glm.ortho(-10, 10, -10, 10, near_plane, far_plane)
        light_view = glm.lookAt(light_pos, glm.vec3(0), glm.vec3(0, 1, 0))
        self.lightSpaceMatrix = light_projection * light_view

    def create(self, width, height):
        self.width = width
        self.height = height
        self.depth_buffer = DepthBuffer()
        self.frame_buffer = FrameBuffer()
        self.frame_buffer.bind()
        self.depth_buffer.create(width, height)
        self.depth_buffer.attach()
        self.frame_buffer.check_complete()

    def cast_shadow(self, depth_program):
        glDisable(GL_CULL_FACE)
        depth_program.use()
        depth_program.setMat4('lightSpaceMatrix', self.lightSpaceMatrix)
        glViewport(0, 0, self.width, self.height)
        self.frame_buffer.bind()
        glClear(GL_DEPTH_BUFFER_BIT)

    def end_cast_shadow(self, program):
        self.frame_buffer.unbind()
        program.use()
        program.setMat4('lightSpaceMatrix', self.lightSpaceMatrix)
        program.setInt('shadowMap', 10)
        glActiveTexture(GL_TEXTURE10)
        self.depth_buffer.bind()
        glEnable(GL_CULL_FACE)

    def delete(self):
        self.frame_buffer.delete()
        self.depth_buffer.delete()
