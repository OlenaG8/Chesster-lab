import os

dataset = "../models/cnn/dataset/6-classes/white_rook_on_white/"

for i in range(0, 100):
    if (os.path.exists(f'{dataset}{i}.png')):
        os.rename(f'{dataset}{i}.png', f'{dataset}wrw{i}.png')