import os
from PIL import  Image

input_path = '/home/zhangjie/pix2pix/face2face_2times/images'
output_path = '/home/zhangjie/pix2pix/trainDataStage2/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

for img_name in os.listdir(input_path):
    name = os.path.basename(img_name)
    name, _ = os.path.splitext(name)
    name1 = name.split('-')[0]
    name2 = name.split('-')[1]

    print(name)
    print(name1)
    print(name2)
    if name2 != 'outputs':
        continue
    img = Image.open(os.path.join(input_path, img_name))
    img.save(output_path + name1 + '.png')