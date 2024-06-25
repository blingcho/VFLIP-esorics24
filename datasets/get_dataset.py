from datasets import cifar10, cinic10, imagenette
import torchvision.datasets as datasets

def get_dataset_by_name(dataset_name):
    dict_dataset = {
        'CIFAR10': datasets.CIFAR10,
        'CINIC10L': cinic10.CINIC10L,
        'Imagenette': imagenette.Imagenette
    }
    dataset = dict_dataset[dataset_name]
    return dataset

def get_datasets_for_ssl(dataset_name, file_path, n_labeled, party_num=None):
    dataset_setup = get_dataset_setup_by_name(dataset_name)
    train_labeled_set, train_unlabeled_set, test_set, train_complete_dataset = \
        dataset_setup.set_datasets_for_ssl(file_path, n_labeled, party_num)
    return train_labeled_set, train_unlabeled_set, test_set, train_complete_dataset

def get_dataset_setup_by_name(dataset_name):
    dict_dataset_setup = {
        'CIFAR10': cifar10.Cifar10Setup(),
        'CINIC10L': cinic10.Cinic10LSetup(),
        'Imagenette': imagenette.ImagenetteSetup()
    }
    dataset_setup = dict_dataset_setup[dataset_name]
    return dataset_setup
