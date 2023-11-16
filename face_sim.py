import argparse
import torch
from PIL import Image
import numpy as np
from score_utils.face_model import FaceAnalysis


def read_img_pil(p):
    return Image.open(p).convert("RGB")

def pil_to_cv2(pil_img):
    return np.array(pil_img)[:,:,::-1]

def get_face_embedding(face_model, img):
    """ get face embedding
    """
    if type(img) is not np.ndarray:
        img = pil_to_cv2(img)
        
    faces = face_model.get(img, max_num=1) ## only get first face
    if len(faces) <= 0:
        return None
    else:
        emb = torch.Tensor(faces[0]['embedding']).unsqueeze(0)
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str)
    parser.add_argument('--img2', type=str)
    args = parser.parse_args()

    img1 = read_img_pil(args.img1)
    img2 = read_img_pil(args.img2)

    face_model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_model.prepare(ctx_id=0, det_size=(512, 512))

    face_emb1 = get_face_embedding(face_model, img1)
    face_emb2 = get_face_embedding(face_model, img2)

    print((face_emb1 @ face_emb2.T).mean().item())
