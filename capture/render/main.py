import torch
import numpy as np

from .scene import Light, Scene, Camera, dot_product, normalize, generate_normalized_random_direction, gamma_encode


class Renderer:
    def __init__(self, return_params=False):
        self.use_augmentation = False
        self.return_params = return_params

    def xi(self, x):
        return (x > 0.0) * torch.ones_like(x)

    def compute_microfacet_distribution(self, roughness, NH):
        alpha = roughness**2
        alpha_squared = alpha**2
        NH_squared = NH**2
        denominator_part = torch.clamp(NH_squared * (alpha_squared + (1 - NH_squared) / NH_squared), min=0.001)
        return (alpha_squared * self.xi(NH)) / (np.pi * denominator_part**2)

    def compute_fresnel(self, F0, VH):
        # https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
        return F0 + (1.0 - F0) * (1.0 - VH)**5

    def compute_g1(self, roughness, XH, XN):
        alpha = roughness**2
        alpha_squared = alpha**2
        XN_squared = XN**2
        return 2 * self.xi(XH / XN) / (1 + torch.sqrt(1 + alpha_squared * (1.0 - XN_squared) / XN_squared))

    def compute_geometry(self, roughness, VH, LH, VN, LN):
        return self.compute_g1(roughness, VH, VN) * self.compute_g1(roughness, LH, LN)

    def compute_specular_term(self, wi, wo, albedo, normals, roughness, metalness):
        F0 = 0.04 * (1. - metalness) + metalness * albedo

        # Compute the half direction
        H = normalize((wi + wo) / 2.0)

        # Precompute some dot product
        NH = torch.clamp(dot_product(normals, H), min=0.001)
        VH = torch.clamp(dot_product(wo, H), min=0.001)
        LH = torch.clamp(dot_product(wi, H), min=0.001)
        VN = torch.clamp(dot_product(wo, normals), min=0.001)
        LN = torch.clamp(dot_product(wi, normals), min=0.001)

        F = self.compute_fresnel(F0, VH)
        G = self.compute_geometry(roughness, VH, LH, VN, LN)
        D = self.compute_microfacet_distribution(roughness, NH)

        return F * G * D / (4.0 * VN * LN)

    def compute_diffuse_term(self, albedo, metalness):
        return  albedo * (1. - metalness) / np.pi

    def evaluate_brdf(self, wi, wo, normals, albedo, roughness, metalness):
        diffuse_term = self.compute_diffuse_term(albedo, metalness)
        specular_term = self.compute_specular_term(wi, wo, albedo, normals, roughness, metalness)
        return diffuse_term, specular_term

    def render(self, scene, svbrdf):
        normals, albedo, roughness, displacement = svbrdf
        device = albedo.device

        # Generate surface coordinates for the material patch
        # The center point of the patch is located at (0, 0, 0) which is the center of the global coordinate system.
        # The patch itself spans from (-1, -1, 0) to (1, 1, 0).
        xcoords_row = torch.linspace(-1, 1, albedo.shape[-1], device=device)
        xcoords = xcoords_row.unsqueeze(0).expand(albedo.shape[-2], albedo.shape[-1]).unsqueeze(0)
        ycoords = -1 * torch.transpose(xcoords, dim0=1, dim1=2)
        coords = torch.cat((xcoords, ycoords, torch.zeros_like(xcoords)), dim=0)

        # We treat the center of the material patch as focal point of the camera
        camera_pos = scene.camera.pos.unsqueeze(-1).unsqueeze(-1).to(device)
        relative_camera_pos = camera_pos - coords
        wo = normalize(relative_camera_pos)

        # Avoid zero roughness (i. e., potential division by zero)
        roughness = torch.clamp(roughness, min=0.001)

        light_pos = scene.light.pos.unsqueeze(-1).unsqueeze(-1).to(device)
        relative_light_pos = light_pos - coords
        wi = normalize(relative_light_pos)

        fdiffuse, fspecular  = self.evaluate_brdf(wi, wo, normals, albedo, roughness, metalness=0)
        f = fdiffuse + fspecular

        color = scene.light.color if torch.is_tensor(scene.light.color) else torch.tensor(scene.light.color)
        light_color = color.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        falloff     = 1.0 / torch.sqrt(dot_product(relative_light_pos, relative_light_pos))**2 # Radial light intensity falloff
        LN = torch.clamp(dot_product(wi, normals), min=0.0) # Only consider the upper hemisphere
        radiance    = torch.mul(torch.mul(f, light_color * falloff), LN)

        return radiance

    def _get_input_params(self, n_samples, light, pose):
        min_eps = 0.001
        max_eps = 0.02
        light_distance = 2.197
        view_distance = 2.75

        # Generate scenes (camera and light configurations)
        # In the first configuration, the light and view direction are guaranteed to be perpendicular to the material sample.
        # For the remaining cases, both are randomly sampled from a hemisphere.
        view_dist = torch.ones(n_samples-1) * view_distance
        if pose is None:
            view_poses = torch.cat([torch.Tensor(2).uniform_(-0.25, 0.25), torch.ones(1) * view_distance], dim=-1).unsqueeze(0)
            if n_samples > 1:
                hemi_views = generate_normalized_random_direction(n_samples - 1, min_eps=min_eps, max_eps=max_eps) * view_distance
                view_poses = torch.cat([view_poses, hemi_views])
        else:
            assert torch.is_tensor(pose)
            view_poses = pose.cpu()

        if light is None:
            light_poses = torch.cat([torch.Tensor(2).uniform_(-0.75, 0.75), torch.ones(1) * light_distance], dim=-1).unsqueeze(0)
            if n_samples > 1:
                hemi_lights = generate_normalized_random_direction(n_samples - 1, min_eps=min_eps, max_eps=max_eps) * light_distance
                light_poses = torch.cat([light_poses, hemi_lights])
        else:
            assert torch.is_tensor(light)
            light_poses = light.cpu()

        light_colors = torch.Tensor([10.0]).unsqueeze(-1).expand(n_samples, 3)

        return view_poses, light_poses, light_colors

    def __call__(self, svbrdf, n_samples=1, lights=None, poses=None):
        view_poses, light_poses, light_colors = self._get_input_params(n_samples, lights, poses)

        renderings = []
        for wo, wi, c in zip(view_poses, light_poses, light_colors):
            scene = Scene(Camera(wo), Light(wi, c))
            rendering = self.render(scene, svbrdf)

            # Simulate noise
            std_deviation_noise = torch.exp(torch.Tensor(1).normal_(mean = np.log(0.005), std=0.3)).numpy()[0]
            noise = torch.zeros_like(rendering).normal_(mean=0.0, std=std_deviation_noise)

            # clipping
            post_noise = torch.clamp(rendering + noise, min=0.0, max=1.0)

            # gamma encoding
            post_gamma = gamma_encode(post_noise)

            renderings.append(post_gamma)

        renderings = torch.cat(renderings, dim=0)

        if self.return_params:
            return renderings, (view_poses, light_poses, light_colors)
        return renderings