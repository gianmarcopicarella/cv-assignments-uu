from OpenGL.GL import *
from engine.renderable.quad import Quad


class Bloom:
    def __init__(self, hdr_buffer, hdr_program, blur_buffer, blurProgram):
        self.hdrbuffer = hdr_buffer
        self.hdrProgram = hdr_program
        self.blurbuffer = blur_buffer
        self.blurProgram = blurProgram
        self.quad = Quad()

    def draw_processed_scene(self):
        horizontal, first_iteration = True, True
        horizontal_val, nhorizontal_val = 1, 0
        self.blurProgram.use()
        for i in range(10):
            if horizontal:
                horizontal_val, nhorizontal_val = 1, 0
            else:
                horizontal_val, nhorizontal_val = 0, 1

            self.blurProgram.setInt('horizontal', horizontal)
            glBindFramebuffer(GL_FRAMEBUFFER, self.blurbuffer.FBOs[horizontal_val])
            glActiveTexture(GL_TEXTURE0)
            if first_iteration:
                glBindTexture(GL_TEXTURE_2D, self.hdrbuffer.colorBuffers[1])
                first_iteration = False
            else:
                glBindTexture(GL_TEXTURE_2D, self.blurbuffer.colorBuffers[nhorizontal_val])
                first_iteration = False

            horizontal = not horizontal
            self.quad.draw()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.hdrProgram.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.hdrbuffer.colorBuffers[0])
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.blurbuffer.colorBuffers[nhorizontal_val])
        self.quad.draw()
