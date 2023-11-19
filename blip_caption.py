import os
import re
import glob
import torch
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

# Define the root directory
root_dir = 'train_data/data_provide_images'

for img_path in os.listdir(root_dir):
    if not img_path.endswith('.jpg') or os.path.exists(img_path + '.txt'):
        continue
    # Open and process the image
    img_path = os.path.join(root_dir, img_path)
    raw_image = ImageOps.exif_transpose(Image.open(img_path).convert('RGB'))

    # Conditional image captioning
    inputs = processor(raw_image, text='this is', return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    caption = re.sub(r'^this is\s*','',caption).capitalize()
    print(caption)
    # Write the caption to a .txt file
    with open(img_path + '.txt', 'w') as f:
        f.write(caption)
