import torch
import clip
from PIL import Image

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load('ViT-B/32', device=device)

    text = clip.tokenize(["a boy"]).to(device)

    image_path = './train_data/boy1/1.jpeg'
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Load the noise
    noise = torch.load('noise.pth').to(device)

    # Add noise to the image
    image = image + noise

    with torch.no_grad():
        
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity =  image_features @ text_features.T

        print(f"Similarity is: {similarity.item()}")

if __name__ == "__main__":
    main()