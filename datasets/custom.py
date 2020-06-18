"""
Detectron2 dataset
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import pickle
from PIL import Image
import os

import datasets.transforms as T


class CustomDetection:
    def __init__(self, custom_dataset, transforms):
        self._transforms = transforms
        self.dataset_dicts = self._get_dataset_dicts(custom_dataset)
        self.prepare = Prepare()

    def _get_dataset_dicts(self, dataset_filename):
        with open(dataset_filename, 'rb') as fp:
            return pickle.load(fp)

    def __getitem__(self, idx):
        img, target = self.prepare(self.dataset_dicts[idx])
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.dataset_dicts)


class Prepare(object):
    def __init__(self):
        pass

    def __call__(self, dataset_dict):
        image = Image.open(dataset_dict['file_name']).convert('RGB')
        w, h = image.size

        anno = dataset_dict["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = torch.tensor([dataset_dict['image_id']])

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.custom_dataset_path)
    assert root.exists(), f'provided dataset path {root} does not exist'
    mode = 'instances'
    dataset = CustomDetection(os.path.join(root, image_set, args.custom_dataset),
                              transforms=make_coco_transforms(image_set))
    return dataset
