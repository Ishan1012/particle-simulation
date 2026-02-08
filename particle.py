import numpy as np
import cv2

class Particle:
    def __init__(self, x, y, radius=60):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.random.randn(2) * 0.5
        self.base_color = np.array([255, 150, 150], dtype=float)
        self.color = (255, 150, 150)
        self.lifetime = np.random.randint(80, 120)
        
        r = radius * np.sqrt(np.random.random())
        theta = np.random.random() * 2 * np.pi
        self.target_offset = np.array([r * np.cos(theta), r * np.sin(theta)])

    def update(self, hand_pos=None, scale=1.0, gesture=None):
        if hand_pos is not None:
            target = hand_pos + (self.target_offset * scale)
            direction = target - self.pos
            self.vel += direction * 0.05
            self.vel *= 0.92

            if gesture == "Fist":
                self.color = (100, 100, 255)
            else:
                self.color = (255, 150, 150)
        else:
            self.vel *= 0.95
            
        self.pos += self.vel
        self.lifetime -= 1

class ParticleManager:
    def __init__(self):
        self.particles = []
        self.max_particles = 600

    def spawn(self, x, y, count=10):
        if len(self.particles) < self.max_particles:
            for _ in range(count):
                self.particles.append(Particle(x, y))

    def update_and_draw(self, frame, hand_pos=None, gesture=None):
        h, w = frame.shape[:2]
        
        scale = 1.0
        if gesture == "Open Palm":
            scale = 4.5
        elif gesture == "Fist":
            scale = 0.2

        for p in self.particles[:]:
            p.update(hand_pos, scale=scale, gesture=gesture)
            
            p.pos[0] = np.clip(p.pos[0], 0, w)
            p.pos[1] = np.clip(p.pos[1], 0, h)
            
            alpha = max(0, p.lifetime / 120)
            size = max(1, int(5 * alpha))
            cv2.circle(frame, (int(p.pos[0]), int(p.pos[1])), size, p.color, -1)
            
            if p.lifetime <= 0:
                self.particles.remove(p)