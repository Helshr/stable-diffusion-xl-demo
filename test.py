import gradio as gr

from diffusers import DiffusionPipeline
import torch

import base64
from io import BytesIO
import os
import gc
from datetime import datetime

from share_btn import community_icon_html, loading_icon_html, share_js

# SDXL code: https://github.com/huggingface/diffusers/pull/3859

model_dir = os.getenv("SDXL_MODEL_DIR")
access_token = os.getenv("ACCESS_TOKEN")

if model_dir:
    # Use local model
    model_key_base = os.path.join(model_dir, "stable-diffusion-xl-base-0.9")
    model_key_refiner = os.path.join(model_dir, "stable-diffusion-xl-refiner-0.9")
else:
    model_key_base = "stabilityai/stable-diffusion-xl-base-0.9"
    model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-0.9"

# Use refiner (enabled by default)
enable_refiner = os.getenv("ENABLE_REFINER", "true").lower() == "true"
# Output images before the refiner and after the refiner
output_images_before_refiner = True

# Create public link
share = os.getenv("SHARE", "false").lower() == "true"

print("Loading model", model_key_base)
pipe = DiffusionPipeline.from_pretrained(model_key_base, torch_dtype=torch.float16, use_auth_token=access_token)

#pipe.enable_model_cpu_offload()
pipe.to("cuda")

# if using torch < 2.0
pipe.enable_xformers_memory_efficient_attention()



# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

if enable_refiner:
    print("Loading model", model_key_refiner)
    pipe_refiner = DiffusionPipeline.from_pretrained(model_key_refiner, torch_dtype=torch.float16, use_auth_token=access_token)
    #pipe_refiner.enable_model_cpu_offload()
    pipe_refiner.to("cuda")

    # if using torch < 2.0
    pipe_refiner.enable_xformers_memory_efficient_attention()

    # pipe_refiner.unet = torch.compile(pipe_refiner.unet, mode="reduce-overhead", fullgraph=True)

# NOTE: we do not have word list filtering in this gradio demo



is_gpu_busy = False

def infer(prompt, negative, scale, samples=4, steps=25, refiner_strength=0.3, num_images=1):
    print("prompt: ", prompt)
    print("negative: ", negative)
    print("scale: ", scale)
    print("samples: ", samples)
    print("steps: ", steps)
    print("refiner_strength: ", refiner_strength)
    print("num_images: ", num_images)
    prompt, negative = [prompt] * samples, [negative] * samples
    images_b64_list = []

    for i in range(0, num_images):
        images = pipe(prompt=prompt, negative_prompt=negative, guidance_scale=scale, num_inference_steps=steps).images
        os.makedirs(r"stable-diffusion-xl-demo/outputs", exist_ok=True)
        gc.collect()
        torch.cuda.empty_cache()


        if enable_refiner:
            if output_images_before_refiner:
                for image in images:
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    image_b64 = (f"data:image/jpeg;base64,{img_str}")
                    images_b64_list.append(image_b64)

            images = pipe_refiner(prompt=prompt, negative_prompt=negative, image=images, num_inference_steps=steps, strength=refiner_strength).images

            gc.collect()
            torch.cuda.empty_cache()

        # Create the outputs folder if it doesn't exist


        for i, image in enumerate(images):
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            image_b64 = (f"data:image/jpeg;base64,{img_str}")
            images_b64_list.append(image_b64)
            # Save the image as PNG with unique timestamp
            filename = f"stable-diffusion-xl-demo/outputs/generated_image_{timestamp}_{i}.png"
            image.save(filename, format="PNG")
    print("images_b64_list: ", len(images_b64_list))
    return images_b64_list

prompt = "Pixar style, Disney style, colorful style, Children's Illustration Style,American Illustration Style, Tony,Saturday morning,park,nature,picnic,green t-shirt,shorts,sneakers,parents,car,excitement,discover"
negative = "ugly,low quality, 3d, photorealistic,poorly Rendered face,poorly drawn face,poor facial details,poorly drawn hands,poorly rendered hands,low resolution,bad composition,mutated body parts,blurry image, disfigured, oversaturated,bad anatomy,deformed body features"
scale = 9
infer(prompt, negative, scale)

