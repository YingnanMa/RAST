import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import net_v19
import time


def test_transform():
    transform_list = []
    transform_list.append(transforms.Resize([512,512]))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content_dir', type=str, default = './content/')
parser.add_argument('--style_dir', type=str, default = './style/')
parser.add_argument('--steps', type=str, default = 1)
parser.add_argument('--vgg', type=str, default = 'model/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default = 'model/decoder_iter_160000.pth')
parser.add_argument('--transform', type=str, default = 'model/transformer_iter_160000.pth')

# Additional options
parser.add_argument('--save_ext', default = '.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default = './output/',
                    help='Directory to save the output image(s)')

# Advanced options

args = parser.parse_args()

device = torch.device("cuda")#"cuda" if torch.cuda.is_available() else 

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = net_v19.decoder
transform = net_v19.Transform(in_planes = 512)
vgg = net_v19.vgg

decoder.eval()
transform.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform))
vgg.load_state_dict(torch.load(args.vgg))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)

content_tf = test_transform()
style_tf = test_transform()

nums=len(os.listdir(args.content_dir))
for i in tqdm(range(1,nums+1)):

        content = content_tf(Image.open(args.content_dir+str(i)+".jpg"))
        style = style_tf(Image.open(args.style_dir+str(i)+".jpg"))

        content = content.to(device).unsqueeze(0)
        style = style.to(device).unsqueeze(0)

        with torch.no_grad():

            for x in range(args.steps):

                start_time=time.time()
                Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
                Content5_1 = enc_5(Content4_1)
            
                Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
                Style5_1 = enc_5(Style4_1)
            
                content = decoder(transform(Content4_1, Style4_1, Content5_1, Style5_1))
                end_time=time.time()
                running_time=end_time-start_time
                print("running_time:",running_time)
                content.clamp(0, 255)

            content = content.cpu()
            
            output_name = args.output+str(i)+args.save_ext
            save_image(content, output_name)
