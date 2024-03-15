import math
import torch


def encode_as_unit_interval(tensor):
    """
        Maps range [-1, 1] to [0, 1]
    """
    return (tensor + 1) / 2

def decode_from_unit_interval(tensor):
    """
        Maps range [0, 1] to [-1, 1]
    """
    return tensor * 2 - 1

def gamma_decode(images):
    return torch.pow(images, 2.2)

def gamma_encode(images):
    return torch.pow(images, 1.0/2.2)

def dot_product(a, b):
    return torch.sum(torch.mul(a, b), dim=-3, keepdim=True)

def normalize(a):
    return torch.div(a, torch.sqrt(dot_product(a, a)))

def generate_normalized_random_direction(count, min_eps = 0.001, max_eps = 0.05):
    r1 = torch.Tensor(count, 1).uniform_(0.0 + min_eps, 1.0 - max_eps)
    r2 = torch.Tensor(count, 1).uniform_(0.0, 1.0)

    r   = torch.sqrt(r1)
    phi = 2 * math.pi * r2

    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt(1.0 - r**2)

    return torch.cat([x, y, z], axis=-1)

def generate_random_scenes(count):
    # Randomly distribute both, view and light positions
    view_positions  = generate_normalized_random_direction(count, 0.001, 0.1) # shape = [count, 3]
    light_positions = generate_normalized_random_direction(count, 0.001, 0.1)

    scenes = []
    for i in range(count):
        c = Camera(view_positions[i])
        # Light has lower power as the distance to the material plane is not as large
        l = Light(light_positions[i], [20.]*3)
        scenes.append(Scene(c, l))

    return scenes

def generate_specular_scenes(count):
    # Only randomly distribute view positions and place lights in a perfect mirror configuration
    view_positions  = generate_normalized_random_direction(count, 0.001, 0.1) # shape = [count, 3]
    light_positions = view_positions * torch.Tensor([-1.0, -1.0, 1.0]).unsqueeze(0)

    # Reference: "parameters chosen empirically to have a nice distance from a -1;1 surface.""
    distance_view  = torch.exp(torch.Tensor(count, 1).normal_(mean=0.5, std=0.75))
    distance_light = torch.exp(torch.Tensor(count, 1).normal_(mean=0.5, std=0.75))

    # Reference: "Shift position to have highlight elsewhere than in the center."
    # NOTE: This code only creates guaranteed specular highlights in the orthographic rendering, not in the perspective one.
    #       This is because the camera is -looking- at the center of the patch.
    shift = torch.cat([torch.Tensor(count, 2).uniform_(-1.0, 1.0), torch.zeros((count, 1)) + 0.0001], dim=-1)

    view_positions  = view_positions  * distance_view  + shift
    light_positions = light_positions * distance_light + shift

    scenes = []
    for i in range(count):
        c = Camera(view_positions[i])
        l = Light(light_positions[i], [20, 20.0, 20.0])
        scenes.append(Scene(c, l))

    return scenes

class Camera:
    def __init__(self, pos):
        self.pos = pos
    def __str__(self):
        return f'Camera({self.pos.tolist()})'

class Light:
    def __init__(self, pos, color):
        self.pos   = pos
        self.color = color
    def __str__(self):
        return f'Light({self.pos.tolist()}, {self.color})'

class Scene:
    def __init__(self, camera, light):
        self.camera = camera
        self.light  = light
    def __str__(self):
        return f'Scene({self.camera}, {self.light})'
    @classmethod
    def load(cls, o):
        cam, light, color = o
        return Scene(Camera(cam), Light(light, color))
    def export(self):
        return [self.camera.pos, self.light.pos, self.light.color]
