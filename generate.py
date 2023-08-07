from sample import *
from configs.sample_config import get_config
set_seed(42)

config = get_config()
args = get_args()

config.output_path = args.output_path
config.nnet_path = args.restore_path
config.n_samples = 3
config.n_iter = 1
device = "cuda"

# init models
nnet = UViT(**config.nnet)
print(f'load nnet from {config.nnet_path}')
nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'), False)
autoencoder = libs.autoencoder.get_model(**config.autoencoder)
clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device=device)


clip_text_model.to(device)
autoencoder.to(device)
nnet.to(device)


prompts = json.load(open(args.prompt_path, "r"))
prompts = json.load(open(args.prompt_path, "r"))
for prompt_index, prompt in enumerate(prompts):
    config.prompt = prompt
    print("sampling with prompt:", prompt)
    sample(prompt_index, config, nnet, clip_text_model, autoencoder, device)
