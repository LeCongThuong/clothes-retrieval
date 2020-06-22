import pandas as pd
import numpy as np
from skimage import io
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AT


# PyTorch Dataset

class ClothesDataset(Dataset):
    def __init__(self, paths, mapping_label_id, transform, test=False):
        self.paths = paths
        self.mapping_label_id = mapping_label_id
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = Image.open(path)
        img = np.array(img)

        if type(self.transform) == A.core.composition.Compose:
            img = self.transform(image=img)['image']
        else:
            img = self.transform(img)

        if not self.test:
            folder = path.split('/')[-1].split('~')[-4]
            label = self.mapping_label_id[folder]
            sample = {
                'image': img,
                'label': label
            }
        else:
            sample = {
                'image': img
            }
        return sample


# augmentation pipeline for both train and test

def augmentation(image_size, train=True):
    max_crop = image_size // 5
    if train:
        data_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Compose(
                [
                    A.OneOf([
                        A.RandomRain(p=0.1),
                        A.GaussNoise(mean=15),
                        A.GaussianBlur(blur_limit=10, p=0.4),
                        A.MotionBlur(p=0.2)
                    ]),

                    A.OneOf([
                        A.RGBShift(p=1.0,
                                   r_shift_limit=(-10, 10),
                                   g_shift_limit=(-10, 10),
                                   b_shift_limit=(-10, 10)
                                   ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.3, contrast_limit=0.1, p=1),
                        A.HueSaturationValue(hue_shift_limit=20, p=1),
                    ], p=0.6),

                    A.OneOf([
                        A.CLAHE(clip_limit=2),
                        A.IAASharpen(),
                        A.IAAEmboss(),
                    ]),

                    A.OneOf([
                        A.IAAPerspective(p=0.3),
                        A.ElasticTransform(p=0.1)
                    ]),

                    A.OneOf([
                        A.Rotate(limit=25, p=0.6),
                        A.IAAAffine(
                            scale=0.9,
                            translate_px=15,
                            rotate=25,
                            shear=0.2,
                        )
                    ], p=1),

                    A.Cutout(num_holes=1, max_h_size=max_crop, max_w_size=max_crop, p=0.2)],
                p=1
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            AT.ToTensor()
        ])
    else:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return data_transform
