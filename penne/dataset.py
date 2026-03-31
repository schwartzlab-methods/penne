from typing import Tuple, Union
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision.transforms import v2
from PIL import Image

class InferenceDataset(Dataset):
    def __init__(self, paths: Union[list[str], Tuple[str], str]):
        '''Initialize the LiveCell dataset for validation

        Args:
            paths (list[str]): List of paths or one path to the image directories.
        '''
        super(InferenceDataset, self).__init__()
        self.paths = paths if isinstance(paths, (list, tuple)) else [paths]
        self.images: list[str] = []
        for path in self.paths:
            for file in os.listdir(path):
                if file.endswith((".png", ".jpg", ".jpeg", ".tiff")):
                    self.images.append(os.path.join(path, file))
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
        ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        '''retrieve the item at the specified index for inference

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, str]: The preprocessed image tensor and the image file name.
        '''
        image_path = self.images[idx]
        f_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image, dtype=np.uint16)  # Convert to numpy array with uint16 dtype
        image_tensor = self.transform(image_array)
        if image_tensor.max().item() > 1.0:
            image_tensor = image_tensor / 255.0  # Rescale to [0, 1]
        image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
        return image_tensor, f_name

class TrainingDataset(Dataset):
    ''''
    Generate the Visium and LIVECell dataset class
    The Visium dataset consists of:
    - tissue_img/: a directory of images of the tissue, in patches
    - mtx/: a directory of truncated matrix files, one for each tissue image, in numpy format
        Each contains a long vector of gene expression counts for each spot in the shape of
        (1, num_genes).

    The LIVECell dataset main directory consists of multiple directories,
    each directory is the name of the cell type, and contains images of that cell type.

    Attributes:
        tissue_dir (str): Path to the tissue image directory.
        mtx_dir (str): Path to the matrix directory.
        livecell_dir (str): Path to the LIVECell image directory.
        imgs (np.ndarray): Array of image file names.
        mtxs (np.ndarray): Array of matrix file names.
        livecell_path (list[str]): List of paths to the LIVECell images.
        livecell_classes (list[str]): List of cell type labels for each LIVECell image.
        livecell_class_to_idx (dict[str, int]): Mapping from cell type labels to class indices.
        livecell_class_count_dict (dict[str, int]): Mapping from cell type labels to the number of images in each class.
        livecell_targets (list[int]): List of class indices for each LIVECell image.
        he_transforms (torchvision.transforms.Compose): Transformations for HE images.
        pcm_transforms (torchvision.transforms.Compose): Transformations for PCM images.
        num_genes (int): Number of genes in the dataset.
        num_pcm_classes (int): Number of PCM classes (ie: cell types) in the dataset.
    '''
    def __init__(self, tissue_dir: str, mtx_dir: str, livecell_dir: str, use_mtx: bool = True):
        '''Initialize the VisiumHD and LIVECell dataset.

        Args:
            tissue_dir (str): Path to the tissue image directory.
            mtx_dir (str): Path to the matrix directory.
            livecell_dir (str): Path to the LIVECell image main directory.
            use_mtx (bool): Whether to use the matrix files. If False, a zero tensor will be returned for the matrix.
        '''
        super(TrainingDataset, self).__init__()
        self.tissue_dir = tissue_dir
        self.mtxs = np.array(os.listdir(mtx_dir))
        if use_mtx:
            self.mtx_dir = mtx_dir
        else:
            self.mtx_dir = None
        self.imgs = np.array(os.listdir(tissue_dir))
        self.livecell_path = []
        self.livecell_classes = [] #classes are string labels of the cell types
        self._write_attributes(livecell_dir)
        # transformations
        self.he_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.Resize((256, 256)),
        ])
        self.pcm_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.RandomCrop((256,256)),
            v2.Resize((256, 256)),
        ])
        # get the number of genes from the mtx file
        mtx = np.load(os.path.join(mtx_dir, self.mtxs[0]))
        self.num_genes = mtx.shape[1]
        # get the total number of pcm classes
        self.num_pcm_classes = len(self.livecell_class_count_dict)

    def __len__(self):
        return self.imgs.size
    
    def __getitem__(self, idx: int):
        '''Get the item at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int]: 
                The HE image, exp matrix tensor, LIVECell image, HE image path, and LIVECell class index.
        '''
        name = self.imgs[idx].split(".")[0]
        he_image_path = os.path.join(self.tissue_dir, f"{name}.png")
        if self.mtx_dir:
            mtx_path = os.path.join(self.mtx_dir, f"{name}.npy")
            mtx = np.load(mtx_path)
            mtx_tensor = torch.tensor(mtx).float().view(-1)
        else:
            mtx_tensor = torch.zeros((self.num_genes,), dtype=torch.float32) # if no mtx_dir is provided, return a zero tensor
        image = Image.open(he_image_path).convert('RGB')
        # put in tensor
        image = self.he_transforms(image)
        if image.max() > 1:
            image = image / 255 # rescale to [0,1]
        # select a random image from the livecell dataset
        livecell_path = self.livecell_path[idx % len(self.livecell_path)]
        livecell_img = Image.open(livecell_path).convert('RGB')
        livecell_img = self.pcm_transforms(livecell_img)
        if livecell_img.max() > 1:
            livecell_img = livecell_img / 255 # scale to [0,1]
        return image, mtx_tensor, livecell_img, he_image_path, self.livecell_class_to_idx[self.livecell_classes[idx % len(self.livecell_classes)]]
    
    @staticmethod
    def _find_all_files(path):
        '''Find all files in a directory.

        Args:
            path (str): Path to the directory.

        Returns:
            list[str]: List of file paths.
        '''
        all_files = []
        for root, _, files in os.walk(path):
            for file in files:
                all_files.append(os.path.join(root, file))
        return all_files
    
    def _write_attributes(self, livecell_dir):
        '''Write attributes for the LIVECell dataset.

        Args:
            livecell_dir (list[str]): List of paths to the LIVECell directories.
        '''
        for path in livecell_dir:
            all_cls = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
            for cls in all_cls:
                imgs = [os.path.join(root, img) for root, _, imgs in os.walk(os.path.join(path, cls)) for img in imgs]
                self.livecell_path.extend(imgs)
                self.livecell_classes.extend([cls]*len(imgs))
        # get the class to idx mapping
        self.livecell_class_to_idx = {cls: i for i, cls in enumerate(np.unique(self.livecell_classes).tolist())}
        self.livecell_targets = [self.livecell_class_to_idx[x] for x in self.livecell_classes] # targets are the class indices
        assert len(self.livecell_path) == len(self.livecell_targets) == len(self.livecell_classes)
        self.livecell_class_count_dict = {k: self.livecell_classes.count(k) for k in np.unique(self.livecell_classes)}
