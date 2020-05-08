from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import glob
import os.path as osp
import os
import random

"""
Posibles mejoras:
- Comprobar image format es válido --> buscar en internet
- Comprobar parámetros de entrada
- get_image_name_wo_ext podría convertirse en una lambda
- Comprobar que hay imágenes en la carpeta xD
"""

class BaseDataset(Dataset):
    def __init__(self,
                 input_img_dir: str,
                 output_json_dir: str,
                 img_format: str = "png",
                 create_output_dir: bool = True,
                 transform = None) -> None:
        if not osp.isdir(input_img_dir):
            raise NotADirectoryError(f"{input_img_dir} is not a valid input image directory")
        self.input_img_dir = input_img_dir
        if not osp.isdir(output_json_dir) and not create_output_dir:
            raise NotADirectoryError(f"{output_json_dir} directory does not exist")
        elif not osp.isdir(output_json_dir) and create_output_dir:
            os.makedirs(output_json_dir)
        self.output_json_dir = output_json_dir
        self.img_format = img_format
        self.input_img_list = []
        self.output_json_files_list = []
        self.transform = transform

    @staticmethod
    def get_image_name_without_extension(img_filename):
        """
        Extracts image name without extension from file name
        :param img_filename:
        :return:
        """
        return osp.basename(img_filename).split('.')[0]

    @staticmethod
    def get_json_filename_for_image(image_name):
        return image_name + '.json'

    def generate_io_samples_pairs(self):
        self.input_img_list = glob.glob(osp.join(self.input_img_dir, "*." + self.img_format))
        for image_file in self.input_img_list:
            image_name = self.get_image_name_without_extension(image_file)
            self.output_json_files_list.append(osp.join(self.output_json_dir,
                                                        self.get_json_filename_for_image(image_name)))

    def __len__(self):
        return len(self.input_img_list)

    def show_image_and_corresponding_json(self, idx):
        print(f"Image file: {self.input_img_list[idx]}")
        print(f"JSON file: {self.output_json_files_list[idx]}")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_filename = self.input_img_list[idx]
        img = Image.open(img_filename)
        if self.transform:
            img = self.transform(img)
        return img, idx
if __name__ == "__main__":
    dataset = BaseDataset("/media/jld/DATOS_JLD/datasets/cityscapes/train/", "/media/jld/DATOS_JLD/gitrepos/paper-keypoints/train/")
    dataset.generate_io_samples_pairs()
    for _ in range(100):
        idx = random.randint(0, len(dataset))
        print(f"Showing id {idx}")
        dataset.show_image_and_corresponding_json(idx)

