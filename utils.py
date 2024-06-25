from datasets import get_dataset
import torch

def list_of_ints(arg):
        return list(map(int, arg.split(',')))

def set_loaders(args):
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
    zip_ = get_dataset.get_datasets_for_ssl(dataset_name=args.dataset,file_path=args.path_dataset,n_labeled=args.n_labeled,party_num=args.party_num)
    train_labeled_set, train_unlabeled_dataset, test_set, train_complete_dataset = zip_
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_unlabeled_dataset,
        batch_size=args.batch_size,
    )
    labeled_train_loader = torch.utils.data.DataLoader(
        dataset=train_labeled_set,
        batch_size=args.batch_size, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
    )
    
    return train_loader, labeled_train_loader, test_loader, train_complete_dataset, dataset_setup.num_classes

def split_data(data,dataset,party_num):
    if dataset in ['CIFAR10', 'CINIC10L', 'Imagenette']:
        splitted_data = []
        width = int(data.shape[3] / party_num)
        for i in range(party_num):
            if i == party_num - 1:
                splitted_data.append(data[:,:,:,width*i:])    
            else:
                splitted_data.append(data[:,:,:,width*i:width*(i+1)])
    else:
        raise Exception('Unknown dataset name!')
    return splitted_data

def correct_counter(output, target, topk=(1, 5)):
    correct_counts = []
    for k in topk:
        confi, pred = output.topk(k, 1, True, True)
        correct_k = torch.eq(pred, target.view(-1, 1)).sum().float().item()
        correct_counts.append(correct_k)
    return correct_counts