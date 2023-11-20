import torch

model_path = 'model.pth'
emb_path = 'emb.pth'

model_dict = torch.load(model_path)
emb = torch.load(emb_path)
model_dict['emb_kv'] = emb
torch.save(model_dict, model_path)
