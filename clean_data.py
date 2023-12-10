# %%
import os
from libs.autocrop import get_crop_face_func

data_root = 'score_utils'

files = os.listdir(data_root)
out_files = []
# %%
from score_utils.face_model import FaceAnalysis
import numpy as np
from PIL import Image, ImageOps


face_model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_model.prepare(ctx_id=0, det_size=(512, 512))


#%%
def crop_image_to_center(pil_image: Image.Image, bbox: np.array):
    bbox = bbox.astype(int)
    # Calculate the center of the bbox
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2

    # Calculate the left, top, right and bottom coordinates for the new image
    left = bbox_center_x - 256
    top = bbox_center_y - 256
    right = bbox_center_x + 256
    bottom = bbox_center_y + 256
    
    if left<0 or right>512:
        left = 0 
        right = 512
        
    if top<0 or bottom>512:
        top = 0
        bottom = 512


    # Crop the image
    cropped_image = pil_image.crop((left, top, right, bottom))

    # Resize the image to 512x512 if it's not already of that size
    if cropped_image.size != (512, 512):
        cropped_image = cropped_image.resize((512, 512), Image.ANTIALIAS)


    return cropped_image

def face_score(face_model, image):
    bboxes, kpss = face_model.det_model.detect(np.array(image)[:,:,::-1], max_num=1, metric='default')
    if bboxes.shape[0] != 1:
        return None
    score = bboxes[0, 4]
    if score<0.85:
        return None
    
    return crop_image_to_center(image,bboxes[0,:4])
# %%
# check face
from libs.autocrop import get_crop_face_func
import tqdm

face_scores = {}
out_files = {}

save_dir = 'score_utils/1'
os.makedirs(save_dir, exist_ok=True)

for file in tqdm.tqdm(files):
    image_path = os.path.join(data_root,file)

    try:
        pil_image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    except:
        files.remove(file)
        continue
    x,y = pil_image.size
    if x//3>y or y//3 >x or min(x,y)<512:
        files.remove(file)
        continue


    desired_size = 512
    ratio = desired_size / min(pil_image.size)
    new_size = tuple([int(x * ratio) for x in pil_image.size])

    pil_image.thumbnail(new_size, Image.ANTIALIAS)
    image = face_score(face_model,pil_image)
    if image:
        save_path = os.path.join(save_dir, file)
        image.save(save_path)


