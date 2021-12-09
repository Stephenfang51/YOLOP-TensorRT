import torch
import struct
import sys
from utils.torch_utils import select_device
import os

# Initialize
device = select_device('cpu')
# pt_file = sys.argv[1]
pt_file = "/home/liwei.fang/YOLOP-main/weights/End-to-end.pth"
# Load model
# model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32
# model.to(device).eval()

model_dict = torch.load(pt_file, map_location=device)['state_dict']
dirname = os.path.dirname(pt_file)
# with open(os.path.join(dirname,'output_keys.txt'), 'w') as f:
#     for k, v in model_dict.items():
#         f.write('{}, shape:{}\n'.format(k, v.shape))
# print(model_dict.keys())
with open(os.path.join(dirname,'output.wts'), 'w') as f:
    f.write('{}\n'.format(len(model_dict.keys())))
    for k, v in model_dict.items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')


# with open(pt_file.split('.')[0] + '.wts', 'w') as f:
#     f.write('{}\n'.format(len(model.state_dict().keys())))
#     for k, v in model.state_dict().items():
#         vr = v.reshape(-1).cpu().numpy()
#         f.write('{} {} '.format(k, len(vr)))
#         for vv in vr:
#             f.write(' ')
#             f.write(struct.pack('>f',float(vv)).hex())
#         f.write('\n')
