import torch
import torch.nn as nn
from models import model_sets_eval

class VflFramework(nn.Module):

    def __init__(self, args):
        super(VflFramework, self).__init__()
        
        # loss funcs
        self.loss_func_top_model = nn.CrossEntropyLoss()
        #bottom model
        if args.dataset in ["CIFAR10","CINIC10L","Imagenette",'NUS-WIDE']:
            input_dim = 64
            if args.party_num == 8:
                input_dim = 32

        self.input_dim = input_dim
        self.bottom_models = []
        for i in range(args.party_num):
            if i in args.bkd_adversary:
                self.bottom_models.append(model_sets_eval.BottomModel(dataset_name=args.dataset).get_model(
                half=args.half,
                is_adversary=True,
                output_dim = input_dim,party_num=args.party_num,args=args
                ).to(args.device))
            else:   
                self.bottom_models.append(model_sets_eval.BottomModel(dataset_name=args.dataset).get_model(
                half=args.half,
                is_adversary=False,
                output_dim = input_dim,party_num=args.party_num,args=args
                ).to(args.device))
        self.bottom_models = nn.ModuleList(self.bottom_models)

        # top model
        self.top_model = model_sets_eval.TopModel(dataset_name=args.dataset).get_model(input_dim=int(input_dim*args.party_num),args=args)
        self.bottom_model_output_dim = input_dim
        # feat model
        self.pattern_raw = []
        for adv_idx in args.bkd_adversary:
            self.pattern_raw.append(torch.normal(mean=0,std=1e-1,size=[int(self.top_model.fc1top.in_features/args.party_num)]))
        
        self.mask_raw = []
        for adv_idx in args.bkd_adversary:
            self.mask_raw.append(torch.normal(mean=0,std=1e-1,size=[int(self.top_model.fc1top.in_features/args.party_num)]))

        self.fixed_pattern_raw = torch.tensor([1.,-1.,1.,-1.,1.,1.,-1.,1.,-1.,1.] ).to(args.device)
        
        if args.defense_type=='VFLIP':
            self.MAE = model_sets_eval.MAE(input_dim=args.party_num*input_dim,args=args)
            self.MAE.bottom_output_dim = self.input_dim
