import torch
import torch.nn as nn
import json
import os
from huggingface_hub import hf_hub_download
from trellis import models
from trellis.pipelines import samplers
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

class TrellisImageTo3DPipeline:
    def __init__(self, path: str):
        # 1. load config
        config_file = f"{path}/pipeline.json" if os.path.exists(f"{path}/pipeline.json") \
                      else hf_hub_download(path, "pipeline.json")
        with open(config_file) as f:
            args = json.load(f)['args']

        # 2. load models explicitly
        self.models = {}
        for k, v in args['models'].items():
            try:
                self.models[k] = models.from_pretrained(f"{path}/{v}")
            except:
                self.models[k] = models.from_pretrained(v)
            self.models[k].eval()

        # 3. load samplers explicitly
        self.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        self.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        self.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        self.slat_sampler_params = args['slat_sampler']['params']

        # 4. normalization params
        self.slat_normalization = args['slat_normalization']

        # 5. image conditioning model (DINOv2)
        self._init_image_cond_model(args['image_cond_model'])

        self.rembg_session = None
        self.device = 'cuda'

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #what this does is essentially subtract the mean and divide by std
        ]) #this is the imagenet normalization scheme
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Crops the image using alpha channel and remove background
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    def encode_image(self, image):
        """
        1. normalizes the image
        2. apply DINO features
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        image = self.image_cond_model_transform(image) #(b, c, h, w)
        
        #apply dino v2 features
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm'] #(b, seq_leng, features)
        patchtokens = F.layer_norm(features, features.shape[-1:])
        neg_cond = torch.zeros_like(patchtokens)
        inp = {"cond": patchtokens,
                 "neg_cond": neg_cond}
        return inp
        # coords = self.sample_sparse_structure(inp, num_samples, sparse_structure_sampler_params)
    
    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution

        #start from noise distribution (b, features, res,res,res)
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device) 
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample( #sample from ode network, while conditioning on the vit features
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder'] #(b, 1, res,res,res)
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int() #gives u the 3D coordinates where the feature at that voxel is not empty.

        return coords

    def run(self, image):
        image = self.preprocess_image(image)
        inp = self.encode_image([image]) #(B, seq_len, feat_dim)
        coords = self.sample_sparse_structure(inp, num_samples, sparse_structure_sampler_params)



if __name__ == "__main__":
    trellis = TrellisImageTo3DPipeline("microsoft/TRELLIS-image-large")
    trellis.to(torch.device("cuda"))
    print("model weights loaded successfully!")
    image = Image.open("assets/example_image/T.png")
    trellis.run(image)

   #Takes an input image of shape (H,W,3)