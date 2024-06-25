import argparse
import os
import dill
import random
import sys
sys.path.insert(0, "./")

import torch
import torch.backends.cudnn as cudnn
from models.vflFramework import VflFramework
from utils import set_loaders, list_of_ints
from eval import evaluation

def main(args):
    _, _, test_loader, _, num_classes = set_loaders(args)
    args.num_classes = num_classes
    model = VflFramework(args)
    model = model.to(args.device)
    cudnn.benchmark = True
    dir_save_model = args.load_dir
    temp = 'ALL' if not args.defense_type=='DPSGD' else 'DPSGD'
    model_name = f"{args.dataset}_{args.attack_type}_{temp}_{args.active}_{args.party_num}" 
    model_file_name = f"{model_name}.pth"
    bkd_pattern_file_name = f"{model_name}_pattern.pt"
    
    model_save_path = os.path.join(dir_save_model,model_file_name)
    bkd_pattern_path = os.path.join(dir_save_model,bkd_pattern_file_name)
    
    if args.attack_type=='VILLAIN':
        print(f"load backddor pattern from {bkd_pattern_path}")
        backdoor_checkpoint=torch.load(bkd_pattern_path,pickle_module=dill)
        model.pattern_raw = backdoor_checkpoint['pattern_raw']
        model.mask_raw = backdoor_checkpoint['mask_raw']

    print(f"load model from {model_save_path}")
    checkpoint = torch.load(model_save_path,pickle_module=dill)

    model.bottom_models.load_state_dict(checkpoint['model_bottom'])
    model.top_model.load_state_dict(checkpoint['model_top'])

    if args.defense_type=='VFLIP' : 
        model.MAE.load_state_dict(checkpoint['mae']); 
        model.MAE.mu = checkpoint['mae_mu'].to(device)
        model.MAE.std = checkpoint['mae_std'].to(device)
        model.MAE.s_score_mean = checkpoint['s_score_mean']
        model.MAE.s_score_std = checkpoint['s_score_std']
        model.MAE.train_mode=False
        threshold_list = [2.0,1.5]

        print(f"\n=== without defense ===")
        args.defense_type='NONE'
        evaluation(args,test_loader,model)

        print(f"\n===with VFLIP===\n")
        args.defense_type='VFLIP'
        for th in threshold_list :
            model.MAE.threshold=th
            model.MAE.s_score_threshold = model.MAE.s_score_mean + model.MAE.threshold* model.MAE.s_score_std
            print(f"---VFLIP threshold {model.MAE.threshold}---")
            evaluation(args,test_loader,model)
    else:
        evaluation(args,test_loader,model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='vfl framework training')
    # dataset paras
    parser.add_argument('-d', '--dataset', default='Criteo', type=str,
                        help='name of dataset',
                        choices=['CIFAR10', 'CINIC10L', 'Imagenette', 'nus_wide'])
    parser.add_argument('--path-dataset', help='path_dataset',
                        type=str, default='./data')
    # model path params
    parser.add_argument('--load-dir', dest='load_dir',
                        help='The directory that have the trained models and pattern files',
                        default='./toy_data', type=str)
    # framework params
    parser.add_argument('--k', help='top k accuracy',
                        type=int, default=4)
    parser.add_argument('--n-labeled',  
                        help='lia label',
                        type=int, default=200)
    parser.add_argument('--party-num',  
                        help='party num',
                        type=int, default=2)
    # attack params
    parser.add_argument('--attack-type',
                        help='The attack type',
                        choices=['VILLAIN', 'BadVFL'], 
                        default='VILLAIN', type=str)
    parser.add_argument('--active',  
                        help='active',
                          action='store_true', default=False)
    parser.add_argument('--bkd-adversary',  
                        help='bkd adversary',
                        type=list_of_ints, default=[1])
    parser.add_argument('--bkd-label',  
                        help='bkd label',
                        type=int, default=0)
    # defense params
    parser.add_argument('--defense-type',
                        help='The defense type',
                        choices=['VFLIP', 'DPSGD', 'NONE'], 
                        default='VFLIP', type=str)

    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--gpu', default=0, type=int, 
                        help='GPU for cuda')
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.half = 16

    seed_main = 1234
    torch.manual_seed(seed_main)         
    torch.cuda.manual_seed(seed_main)    
    random.seed(seed_main)

    # pht name : {dataset}_{attack_type}_{defense_type}_{active}_{party_num}.pth
    main(args)
