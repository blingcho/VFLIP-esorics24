import copy
import torch
import torch.utils.data
import torch.nn.functional as F

from utils import split_data, correct_counter

def add_backdoor(inputs,labels,bkd_proportion,party_num,dataset,bkd_label,device='cpu'):
    attack_portion = round(inputs.shape[0] * bkd_proportion)

    pattern = torch.tensor([
	[1.,1., 0., 1.,1.],
        [0.,1., 1., 1.,0.],
        [0.,0., 1., 0.,0.],
        [0.,1., 1., 1.,0.],
	[1.,1., 0., 1.,1.],
    ])
    if party_num == 8:
        pattern = torch.tensor([
            [1.,1., 0., 1.],
            [0.,1., 1., 1.],
            [0.,0., 1., 0.],
            [0.,1., 1., 1.],
            [1.,1., 0., 1.],
            ])
    x_top = 5
    y_top = 0
    if party_num == 4:
        x_top = 18
        y_top = 2
    if dataset in ["Imagenette"]:
        if party_num == 8:
            pattern = torch.tensor([
            [1.,1., 0., 1.,1.],
                [0.,1., 1., 1.,0.],
                [0.,0., 1., 0.,0.],
                [0.,1., 1., 1.,0.],
            [1.,1., 0., 1.,1.],
            ])
            pattern = pattern.repeat(5,5)
        else:
            pattern = pattern.repeat(8,8)

        x_top = 100
        y_top = 2
    elif dataset in ["nus_wide"]:
        pattern = torch.tensor([[1.,-1.,1.,-1.,1.,-1.,1.,-1.,-1.,1.,-1.,1.,-1.,1.,-1.,1.,-1.,1.,-1.,1.,-1.,1.,-1.,-1.,1.,-1.,1.,-1.,1.,-1.,1.,-1.,1.,-1.,1.,-1.,1.,-1.,-1.,1. ,1.,-1.,1.,-1.,1.,-1.,1.,-1.,-1.,1.,1.,-1.,1.,-1.,1.,-1.,1.,-1.,-1.,1.  ]])*5
        x_start = 30
        x_end = x_start + pattern.shape[1]
    else:
        x_bot = x_top + pattern.shape[0]
        y_bot = y_top + pattern.shape[1]

    mask_value = -10

    full_pattern = torch.zeros(inputs.shape[1:])
    full_pattern.fill_(mask_value)

    if dataset in ["bank", "givemesomecredit"]:
        full_pattern[x_start:x_end] = pattern
    else:
        try:
            full_pattern[:, x_top:x_bot, y_top:y_bot] = pattern
        except:
            print("pattern error")

    mask = 1 * (full_pattern != mask_value)
    mask = mask.to(device)

    bkd_pattern = full_pattern.to(device)
    bkd_inputs = copy.deepcopy(inputs).to(device)
    bkd_labels = copy.deepcopy(labels).to(device)

    bkd_inputs[:attack_portion] = (1 - mask) * inputs[:attack_portion]
    bkd_inputs[:attack_portion,:] += mask * bkd_pattern
    bkd_labels[:attack_portion].fill_(int(bkd_label))

    return bkd_inputs, bkd_labels


def add_villain_backdoor(inputs,labels,bkd_proportion,pattern_raw,mask_raw,adv_idx,bkd_label,device='cpu'):

    pattern = pattern_raw[adv_idx]
    mask = (torch.tanh(mask_raw[adv_idx]) + 1) / 2
    pattern = pattern.to(device)
    mask = mask.to(device)

    attack_portion = round(inputs.shape[0] * bkd_proportion)
    x_top = 0
    x_bot = x_top + pattern.shape[0]
    mask_value = -100

    full_pattern = torch.zeros(inputs.shape[1:])
    full_pattern.fill_(mask_value)
    full_pattern[x_top:x_bot] = pattern

    bkd_pattern = full_pattern
    bkd_pattern = bkd_pattern.to(device)
    bkd_inputs = copy.deepcopy(inputs).to(device)
    bkd_labels = copy.deepcopy(labels).to(device)
    bkd_inputs[:attack_portion] = inputs[:attack_portion]
    bkd_inputs[:attack_portion,:] += mask * bkd_pattern
    bkd_labels[:attack_portion].fill_(int(bkd_label))

    return bkd_inputs, bkd_labels

def bottom_to_top(input_data,framework,party_num,defense_type,device='cpu'):
    bottom_model_output_all = torch.tensor([]).to(device)
    for i in range(party_num):   
        bottom_model_output_all = torch.cat((bottom_model_output_all,input_data[i]), dim=1)
                    
    if defense_type=="VFLIP":
        bottom_model_output_all = framework.MAE(bottom_model_output_all)
        output_framework = framework.top_model(bottom_model_output_all)
    else:
        output_framework = framework.top_model(bottom_model_output_all)
    
    return output_framework

def make_eval(framework, party_num):
    for i in range(party_num):
        framework.bottom_models[i].eval()
    framework.top_model.eval()

def test_per_epoch(args,test_loader, framework):
    test_loss = 0
    correct_top1 = 0
    correct_topk = 0
    count = 0
    k = args.k
    make_eval(framework, args.party_num)
    loss_func_top_model = framework.loss_func_top_model

    for data, target, idx in test_loader:
        data = data.float().to(args.device)
        target = target.long().to(args.device)

        splitted_data= split_data(data,args.dataset,args.party_num)
        input_tensors_top_model = [torch.tensor([], requires_grad=False) for i in range(args.party_num)]
        output_tensors_bottom_model = []

        for i in range(args.party_num):
            output_tensors_bottom_model.append(framework.bottom_models[i](splitted_data[i]).to(args.device))
            input_tensors_top_model[i].data = output_tensors_bottom_model[i].data

        output_framework = bottom_to_top(input_tensors_top_model,framework,args.party_num,args.defense_type,device=args.device)
        correct_top1_batch, correct_topk_batch = correct_counter(output_framework, target, (1, k))
        test_loss += loss_func_top_model(output_framework, target).data.item()
        correct_top1 += correct_top1_batch
        correct_topk += correct_topk_batch
        count += 1

    num_samples = len(test_loader.dataset)
    test_loss /= num_samples
    print('Loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%), Top {} Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss,
        correct_top1, num_samples, 100.00 * float(correct_top1) / num_samples,
        k,
        correct_topk, num_samples, 100.00 * float(correct_topk) / num_samples
    ))


def bkd_test_per_epoch(args, test_loader, framework):
    correct_top1 = 0
    correct_topk = 0
    correct_top1_robust = 0
    count = 0
    num_non_target_samples = 0
    k = args.k
    make_eval(framework, args.party_num)

    for data, target,idx in test_loader:
        data = data.float().to(args.device)
        target = target.long().to(args.device)

        splitted_data= split_data(data,args.dataset,args.party_num)
        input_tensors_top_model = [torch.tensor([], requires_grad=False) for i in range(args.party_num)]
        output_tensors_bottom_model = []
        
        for i in range(args.party_num):
            if i in args.bkd_adversary :
                if args.attack_type in ['BadVFL']:                                      
                    splitted_data[i],target_bkd = add_backdoor(splitted_data[i],target,1.0,args.party_num,args.dataset,args.bkd_label,device=args.device)
                
            output_tensors_bottom_model.append(framework.bottom_models[i](splitted_data[i]))
            
            if i in args.bkd_adversary : 
                if args.attack_type == 'VILLAIN':
                    output_tensors_bottom_model[i].data, target_bkd = add_villain_backdoor(output_tensors_bottom_model[i].data, target,1.0,framework.pattern_raw,framework.mask_raw,args.bkd_adversary.index(i),args.bkd_label,device=args.device)
            input_tensors_top_model[i].data = output_tensors_bottom_model[i].data

        output_framework = bottom_to_top(input_tensors_top_model,framework,args.party_num,args.defense_type,device=args.device)
        
        non_target_label_idx = (target!=args.bkd_label).nonzero(as_tuple=True)[0]
        num_non_target_samples += len(non_target_label_idx)
        correct_top1_batch, correct_topk_batch = correct_counter(output_framework[non_target_label_idx], target_bkd[non_target_label_idx], (1, k))
        
        correct_top1 += correct_top1_batch
        correct_topk += correct_topk_batch
        correct_top1_batch_robust, correct_topk_batch_robust = correct_counter(output_framework, target, (1, k))
        correct_top1_robust += correct_top1_batch_robust
        count += 1

    num_samples = len(test_loader.dataset)
    print('Top 1 ASR: {}/{} ({:.2f}%), Top {} ASR: {}/{} ({:.2f}%), RAC: {:.2f}%\n'.format(
            correct_top1, num_non_target_samples, 100.00 * float(correct_top1) / num_non_target_samples,
            k,
            correct_topk, num_non_target_samples, 100.00 * float(correct_topk) / num_non_target_samples,
            100.00 * float(correct_top1_robust) / num_samples
        ))

def evaluation(args,test_loader,model):
    print('Evaluation on the testing dataset:')
    test_per_epoch(args,test_loader=test_loader, framework=model)
    print('Backdoor Evaluation on the testing dataset:')
    bkd_test_per_epoch(args,test_loader=test_loader, framework=model)