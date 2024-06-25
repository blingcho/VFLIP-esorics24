import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets.dataset_setup import DatasetSetup
from datasets import train_val_split, image_format_2_rgb

class Imagenette(Dataset):
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        image_folder = torchvision.datasets.ImageFolder(root=root + '/' + split)
        self.targets = image_folder.targets
        self.image_paths = image_folder.imgs
        self.transform = transform
    def __getitem__(self, index):
        file_path, label = self.image_paths[index]
        img = self.read_image(file_path)
        return img, label, index
    
    def __len__(self):
        return len(self.image_paths)

    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img


class ImagenetteSetup(DatasetSetup):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.size_bottom_out = 10

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num):
        transforms_ = self.get_transforms()
        base_dataset = Imagenette(file_path, split='train')
        base_test_dataset = Imagenette(file_path, split='val')
        train_labeled_idxs, test_labeled_idxs = train_val_split(base_test_dataset.targets,
                                                                   int(n_labeled / self.num_classes),
                                                                   self.num_classes)
        train_labeled_dataset = ImagenetteLabeled(file_path, train_labeled_idxs, split='val', transform=transforms_)
        train_unlabeled_dataset = ImagenetteLabeled(file_path, None, split='train', transform=transforms_)
        train_complete_dataset = ImagenetteLabeled(file_path, None, split='train', transform=transforms_)
        test_dataset = ImagenetteLabeled(file_path, test_labeled_idxs, split='val', transform=transforms_)
        
        print("#Aux dataset", len(train_labeled_dataset.targets), "#Train:", len(train_unlabeled_dataset.targets),"Test:",len(test_dataset.targets))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset

    def get_normalize_transform(self):
        normalize_imagenette = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return normalize_imagenette

    def get_transforms(self):
        normalize = self.get_normalize_transform()
        transforms_ = transforms.Compose([
            transforms.Lambda(image_format_2_rgb),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        return transforms_

    def get_transformed_dataset(self, file_path, party_num=None, train=True):
        if train:
            split = 'train'
        else:
            split = 'val'
        transforms_ = self.get_transforms()
        _imagenette_dataset = Imagenette(file_path, split, transform=transforms_)
        return _imagenette_dataset

    def clip_one_party_data(self, x, half):
        x = x[:, :, :, :half]
        return x


class ImagenetteLabeled(Imagenette):

    def __init__(self, root, indexs=None, split='train',
                 transform=None):
        super(ImagenetteLabeled, self).__init__(root, split=split,
                                              transform=transform
                                              )
        if indexs is not None:

            random.seed(2024)
            temp_image_paths = []
            for id in indexs:
                temp_image_paths.append(self.image_paths[id])
                
            temp_target = []
            for id in indexs:
                temp_target.append(self.targets[id])
            

            temp = list(zip(temp_image_paths, temp_target))
            random.shuffle(temp)
            self.image_paths,self.targets= zip(*temp)

        else:
            
            random.seed(2024)
            temp_image_paths = []
            indexs = range(len(self.image_paths))
            for id in indexs:
                temp_image_paths.append(self.image_paths[id])
                
            temp_target = []
            for id in indexs:
                temp_target.append(self.targets[id])
            

            temp = list(zip(temp_image_paths, temp_target))
            random.shuffle(temp)
            self.image_paths,self.targets= zip(*temp)

class ImagenetteUnlabeled(ImagenetteLabeled):

    def __init__(self, root, indexs, split='train',
                 transform=None):
        super(ImagenetteUnlabeled, self).__init__(root, indexs, split=split,
                                                transform=transform
                                                )
        temp_image_paths = []
        for image_path, label in self.image_paths:
            temp_image_paths.append((image_path, -1))
        self.image_paths = temp_image_paths


if __name__ == '__main__':
    dataset = Imagenette(root='./data/Imagenette')
    print("s")
