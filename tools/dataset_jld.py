from torch.utils.data import Dataset, DataLoader
import glob

class BaseDataset(Dataset):
    def __init__(self, input_img_dir, output_img_dir) -> None:
        self.input_img_dir = input_img_dir
        self.output_img_dir = output_img_dir
        self.input_img_list = []
    def generate_io_samples_pairs(self):
        self.input_img_list


