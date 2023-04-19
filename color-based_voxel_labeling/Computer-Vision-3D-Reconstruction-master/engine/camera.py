import glm
import math


class Camera:
    def __init__(self, position, pitch=-90, yaw=0, speed=20):
        self.right = None
        self.direction = None
        self.position = position
        self.up = glm.vec3(0, 1, 0)
        self.worldUp = glm.vec3(0, 1, 0)
        self.pitch = pitch
        self.yaw = yaw
        self.speed = speed
        self.sensitivity = 0.25
        self.update_vectors()

    def move_right(self, time):
        self.position += self.right * (self.speed * time)

    def move_left(self, time):
        self.position -= self.right * (self.speed * time)

    def move_top(self, time):
        self.position += self.direction * (self.speed * time)

    def move_bottom(self, time):
        self.position -= self.direction * (self.speed * time)

    def rotate(self, offset_x, offset_y):
        self.yaw += offset_x * self.sensitivity
        self.pitch += offset_y * self.sensitivity
        if self.pitch > 89:
            self.pitch = 89
        elif self.pitch < -89:
            self.pitch = -89
        self.update_vectors()

    def update_vectors(self):
        x = math.cos(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        y = math.sin(glm.radians(self.pitch))
        z = math.sin(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        front = glm.vec3(x, y, z)
        self.direction = glm.normalize(front)
        self.right = glm.normalize(glm.cross(self.direction, self.worldUp))
        self.up = glm.normalize(glm.cross(self.right, self.direction))

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.direction, self.up)
