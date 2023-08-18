import os
import glob
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

# Define the root directory
root_dir = 'train_data'

# Define the subdirectories
sub_dirs = ['boy1', 'boy2', 'girl1', 'girl2']

for sub_dir in sub_dirs:
    # Get list of all .jpeg and .jpg files in the subdirectory
    imgs = glob.glob(os.path.join(root_dir, sub_dir, '*.[jJ][pP]*[gG]'))

    for img_path in imgs:
        # Open and process the image
        raw_image = Image.open(img_path).convert('RGB')

        # Conditional image captioning
        text = "a photo of a " + sub_dir[:-1]
        inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Write the caption to a .txt file
        txt_path = img_path.rsplit('.', 1)[0] + '.txt'
        with open(txt_path, 'w') as f:
            f.write(caption)