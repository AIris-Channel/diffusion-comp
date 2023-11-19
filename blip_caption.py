import os
import re
import random
import torch
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration
import threading


def blip_caption(image_paths, device):
    # Load the model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(device)

    for img_path in image_paths:
        if os.path.exists(img_path + '.txt'):
            continue
        # Open and process the image
        raw_image = ImageOps.exif_transpose(Image.open(img_path).convert('RGB'))

        # Conditional image captioning
        inputs = processor(raw_image, text='this is', return_tensors="pt").to(device, torch.float16)

        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        caption = re.sub(r'^this is\s*','',caption).capitalize()
        print(caption)

        # Write the caption to a .txt file
        with open(img_path + '.txt', 'w') as f:
            f.write(caption)

def random_partition(input_list, num_partitions):
    random.shuffle(input_list)
    list_length = len(input_list)
    partition_size = list_length // num_partitions
    random_partitioned_lists = [input_list[i * partition_size: (i + 1) * partition_size] for i in range(num_partitions)]
    return random_partitioned_lists

if __name__ == '__main__':
    root_dir = 'boy'
    image_paths = [os.path.join(root_dir, i) for i in os.listdir(root_dir) if i.endswith('.jpg')]
    threads = []
    random.shuffle(image_paths)
    threads.append(threading.Thread(target=blip_caption, args=(image_paths, 'cuda:0')))
    random.shuffle(image_paths)
    threads.append(threading.Thread(target=blip_caption, args=(image_paths, 'cuda:1')))
    # threads = []
    # for img_paths in random_partition(image_paths, 8):
    #     threads.append(threading.Thread(target=blip_caption, args=(img_paths,)))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
