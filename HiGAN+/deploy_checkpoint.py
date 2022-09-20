import torch


ckpt_path = './pretrained/HiGAN+.pth'
dst_path = './pretrained/deploy_HiGAN+.pth'

state_dict = torch.load(ckpt_path)
new_state_dict = {}
for key in ['Generator', 'StyleEncoder', 'StyleBackbone']:
    new_state_dict[key] = state_dict[key]

torch.save(new_state_dict, dst_path)
print('save to -> ', dst_path)
