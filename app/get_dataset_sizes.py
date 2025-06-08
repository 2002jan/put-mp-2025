import math
import os
import pickle

from utils.env import Env

env = Env()

md_drives = []
sm_drives = []
xs_drives = []
full_drives = []

for set_name in ['train', 'val']:
    val_f = os.listdir(env.dataset_path / set_name)

    print(len(val_f), set_name)

    images = []

    for f in val_f:
        ff = os.listdir(env.dataset_path / set_name / f / 'image_02' / 'target')
        # print(f, len(ff))
        images.append(len(ff))

    md = math.ceil(sum(images) / 2)
    sm = math.ceil(sum(images) / 4)
    xs = math.ceil(sum(images) / 10)

    images_xs = []
    images_sm = []
    images_md = []

    for f in val_f:
        ff = os.listdir(env.dataset_path / set_name / f / 'image_02' / 'target')

        if sum(images_sm) < sm:
            images_sm.append(len(ff))
            sm_drives.append(f)

        if sum(images_md) < md:
            images_md.append(len(ff))
            md_drives.append(f)

        if sum(images_xs) < xs:
            images_xs.append(len(ff))
            xs_drives.append(f)

        full_drives.append(f)

dataset_types = {
    'full': full_drives,
    'sm': sm_drives,
    'xs': xs_drives,
    'md': md_drives,
}

with open('dataset_types.pickle', 'wb') as pickl:
    pickle.dump(dataset_types, pickl)