
import os
from torch import autocast
from diffusers import StableDiffusionPipeline
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# diffusion
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

# image captioning
img_captioning = pipeline(Tasks.image_captioning, model='damo/ofa_image-caption_coco_large_en')


seed = 'A lovely cat'
os.mkdir(f'diffution_caption/{seed[:50]}')
prompt = seed
step = 1000

for i in range(step):
    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5)["sample"][0]  
        
    caption = img_captioning({'image': image})['caption']
    print(f'step {i}, [original caption]: {prompt}, [diffusion caption]: {caption}')
    image.save(f"diffution_caption/{seed[:50]}/step_{i}_{prompt}.png")

    prompt = caption