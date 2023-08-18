import torch
import torch.nn as nn
import torch.optim as optim
import clip
from PIL import Image

class NoiseGenerator(nn.Module):
    def __init__(self):
        super(NoiseGenerator, self).__init__()
        self.noise = nn.Parameter(torch.randn(3, 224, 224))

    def forward(self, x):
        return x + self.noise

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load('ViT-B/32', device=device)

    # Initialize Noise Generator
    noise_gen = NoiseGenerator().to(device)
    optimizer = optim.Adam(noise_gen.parameters(), lr=0.01)

    text = clip.tokenize(["a boy"]).to(device)

    image_path = './train_data/boy1/1.jpeg'
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    for i in range(1000): # Number of training iterations
        optimizer.zero_grad()

        # Add noise to the image
        noisy_image = noise_gen(image)

        with torch.no_grad():
            text_features = model.encode_text(text)

        # Encode the noisy image
        noisy_image_features = model.encode_image(noisy_image)

        # Normalize the features
        noisy_image_features = noisy_image_features / noisy_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (noisy_image_features @ text_features.T).mean()

        # Maximize the similarity
        loss = -similarity
        loss.backward()
        optimizer.step()

        if i % 100 == 0: 
            print(f"Iteration {i}, Similarity: {similarity.item()}, Loss: {loss.item()}")

            # Save the image
            generated_image = noisy_image[0].detach().cpu()

            # Undo normalization
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            generated_image = generated_image * std + mean

            # Clip values to range [0, 1]
            generated_image = generated_image.clip(0, 1)

            # Transform tensor to image
            generated_image = generated_image.permute(1, 2, 0).numpy()
            generated_image = (generated_image * 255).astype("uint8")

            Image.fromarray(generated_image).save(f"generated_{i}.jpg")
    torch.save(noise_gen.noise, 'noise.pth')
if __name__ == "__main__":
    main()