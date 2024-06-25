import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

from numbers import Number
from torch.autograd import Variable
import numpy as np

D_ = 2 ** 13

def conv3x3(in_features, out_features):
    return nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)

def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def distance(a, b,dist):
    return dist[int(a)][int(b)] + dist[int(b)][int(a)]

def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / bandwidth)) ** 2)

def gaussian_noise(data_shape, s, sigma, device=None):
    return torch.normal(0, sigma * s, data_shape).to(device)

class vgg19(nn.Module):
    def __init__(self,num_classes,n_workers=4,cifar=True,args=None):
        super(vgg19, self).__init__()
        self.n_workers = n_workers
        self.features_1 = nn.Sequential(
            # 1
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 2
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 3
            conv3x3(64, int(128)),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.features_2 = nn.Sequential(
            # 4
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 5
            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 6
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.features_3 = nn.Sequential(
            # 8
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 9
            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 10
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 11
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.features_4 = nn.Sequential(
            #12
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 13
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 14
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 15
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 16
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        if args.dataset in ["BHI"]:
            self.features_3 = nn.Sequential(
                # 8
                conv3x3(256, 256),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                # 9
                conv3x3(256, 512),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                # 10
                conv3x3(512, 512),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                # 11
                conv3x3(512, 512),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                #12
                conv3x3(512, 512),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2),
                            # 13
                conv3x3(512, 512),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                    # 17
                    nn.Linear(4608, 4096),
                    nn.ReLU(),
                    nn.Dropout(),
                    # 18
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Dropout(),
                    # 19
                    nn.Linear(4096, num_classes),
                )

        elif args.dataset in ["Imagenette"]:
            if n_workers == 2:
                self.classifier = nn.Sequential(
                    # 17
                    nn.Linear(3584, 4096),
                    nn.ReLU(),
                    nn.Dropout(),
                    # 18
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Dropout(),
                    # 19
                    nn.Linear(4096, num_classes),
                )
            elif n_workers == 4:
                self.features_4 = nn.Sequential(
                    #12
                    conv3x3(512, 512),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(

                    # 17
                    nn.Linear(7168, 4096),
                    nn.ReLU(),
                    nn.Dropout(),
                    # 18
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Dropout(),
                    # 19
                    nn.Linear(4096, num_classes),
                )
            self.n_workers = 2
        else:
            self.classifier = nn.Sequential(
                # 17
                nn.Linear(2048, 4096),
                nn.ReLU(),
                nn.Dropout(),
                # 18
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(),
                # 19
                nn.Linear(4096, num_classes),
            )
    def forward(self, x):
        out = self.features_1(x)

        if self.n_workers<16:
            out = self.features_2(out)

        if self.n_workers<8:
            out = self.features_3(out)

        if self.n_workers<4:
            out = self.features_4(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    

class MAE(nn.Module):
    def __init__(self,input_dim=20,args=None):
        super(MAE, self).__init__()
        self.args = args
        self.device = args.device
        self.apply(weights_init)
        self.input_dim = input_dim
        self.bottom_output_dim = int(input_dim/args.party_num)
        milestones = [10,40]

        self.MAE_encoder =  nn.Sequential(
                        nn.Linear(self.input_dim, int(self.input_dim/16*15)),
                        nn.ReLU(True),
                        nn.Linear(int(self.input_dim/16*15),int(self.input_dim/8*7)),
                        nn.ReLU(True),
                        nn.Linear(int(self.input_dim/8*7),int(self.input_dim/4*3)),
                        nn.ReLU(True),
                    )
        self.MAE_decoder = nn.Sequential(

                        nn.Linear(int(self.input_dim/4*3),int(self.input_dim/8*7)),
                        nn.ReLU(True),
                        nn.Linear(int(self.input_dim/8*7),  int(self.input_dim/16*15)),
                        nn.ReLU(True),
                        nn.Linear(int(self.input_dim/16*15),   int(self.input_dim)),

                    )
        self.aux_classifier = nn.Sequential(

                        nn.Linear(int(self.input_dim/4*3), int(self.input_dim/2)),
                        nn.ReLU(True),
                        nn.Linear(int(self.input_dim/2), int(args.num_classes)),

                    )
        self.temperature = 1e-1
        self.min = 0
        self.max = 0
        self.clamp = True
        self.train_mode = True
        self.s_score_threshold = 13
        if args.dataset in ['CINIC10L']:
            self.s_score_threshold = 13
        self.threshold = 2.5 if args.dataset in ['Imagenette', 'bank'] else 2.0
        
        self.normalize = True
        self.unit = False
        self.s_score_save_list = []

        self.mu=0
        self.std=1
        
    def forward(self,input_tensor_top_model):        
        if self.train_mode:
            if self.normalize:
                input_tensor_top_model = (input_tensor_top_model - self.mu) / self.std
                orig_input_tensor_top_model = input_tensor_top_model.clone()
            elif self.unit:
                orig_input_tensor_top_model = input_tensor_top_model.clone()
                orig_norm = torch.zeros([orig_input_tensor_top_model.shape[0],self.args.party_num]).to(self.device)
                for i in range(self.args.party_num):
                    orig_norm[:,i] = torch.norm(input_tensor_top_model[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim],dim=1)
                    input_tensor_top_model[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim] = F.normalize(input_tensor_top_model[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim],dim=1)
            latent = self.MAE_encoder(input_tensor_top_model)
            output_bottom_models = self.MAE_decoder(latent)
            if self.normalize:
                output_bottom_models = (output_bottom_models * self.std) + self.mu
            elif self.unit:
                for i in range(self.args.party_num):

                    output_bottom_models[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim] = output_bottom_models[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim]*orig_norm[:,i].unsqueeze(1)
                    input_tensor_top_model[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim] = input_tensor_top_model[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim]*orig_norm[:,i].unsqueeze(1)
        else:
            if self.normalize:
                input_tensor_top_model = (input_tensor_top_model - self.mu) / self.std
                orig_input_tensor_top_model = input_tensor_top_model.clone()
            elif self.unit:
                orig_input_tensor_top_model = input_tensor_top_model.clone()
                orig_norm = torch.zeros([orig_input_tensor_top_model.shape[0],self.args.party_num]).to(self.device)
                for i in range(self.args.party_num):
                    orig_norm[:,i] = torch.norm(input_tensor_top_model[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim],dim=1)
                    input_tensor_top_model[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim] = F.normalize(input_tensor_top_model[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim],dim=1)
                orig_input_tensor_top_model = input_tensor_top_model.clone()
            
            s_score = torch.zeros([input_tensor_top_model.shape[0],self.args.party_num,self.args.party_num])
            for i in range(self.args.party_num):
                mask = torch.zeros_like(input_tensor_top_model).to(self.device)
                mask[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim] = 1
                latent = self.MAE_encoder(mask*input_tensor_top_model)
                recon = self.MAE_decoder(latent)
                for j in range(self.args.party_num):
                    if i == j:
                        continue
                    s_score[:,i,j] = torch.norm(recon[:,j*self.bottom_output_dim:(j+1)*self.bottom_output_dim]-orig_input_tensor_top_model[:,j*self.bottom_output_dim:(j+1)*self.bottom_output_dim],dim=1)
            
            voting_matrix = torch.sum((s_score > self.s_score_threshold).int(),dim=1)
            mask = torch.ones_like(orig_input_tensor_top_model)
            
            for i in range(voting_matrix.shape[0]):
                for j in range(voting_matrix.shape[1]):
                    if voting_matrix[i,j] > int(self.args.party_num/2): 
                        mask[i,j*self.bottom_output_dim:(j+1)*self.bottom_output_dim] = 0
            
            #reconstruction       
            latent = self.MAE_encoder(input_tensor_top_model*mask)
            recon_output_bottom_models = self.MAE_decoder(latent)
            output_bottom_models = recon_output_bottom_models
            if self.normalize:
                output_bottom_models = (output_bottom_models * self.std) + self.mu
            elif self.unit:
                for i in range(self.args.party_num):

                    output_bottom_models[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim] = output_bottom_models[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim]*orig_norm[:,i].unsqueeze(1)
                    input_tensor_top_model[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim] = input_tensor_top_model[:,i*self.bottom_output_dim:(i+1)*self.bottom_output_dim]*orig_norm[:,i].unsqueeze(1)

        return output_bottom_models


class TopModelForCifar10(nn.Module):
    def __init__(self,input_dim=20,args=None):
        super(TopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(input_dim, input_dim)
        self.fc2top = nn.Linear(input_dim, int(input_dim/2))
        self.fc3top = nn.Linear(int(input_dim/2), int(10))
        #self.fc4top = nn.Linear(int(input_dim/2), 10)
        self.args = args
        self.apply(weights_init)
        self.input_dim = input_dim
        if args.dataset in ["CIFAR10","CINIC10L"]:
            self.bottom_output_dim = 10

        elif args.dataset in ["CIFAR100"]:
            self.bottom_output_dim = 100
            
        self.clipper = None
        
    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(std.data.new(std.size()).normal_())

        return mu + eps * std
    
    def forward(self, input_tensor_top_model):
        output_bottom_models = input_tensor_top_model
            
        x = output_bottom_models
        x = self.fc1top(x)
        x = F.relu(x)
        x = self.fc2top(x)
        x = F.relu(x)
        x = self.fc3top(x)
        
        return F.log_softmax(x, dim=1)


class BottomModelForImagenette(nn.Module):
    def __init__(self,output_dim=10,party_num=4,args=None):
        super(BottomModelForImagenette, self).__init__()
        #self.vgg19 = resnet56(num_classes=output_dim)
        self.vgg19 = vgg19(num_classes=output_dim,n_workers= party_num/2,cifar=False,args=args)

    def forward(self, x):
        x = self.vgg19(x)
        return x


class BottomModelForCifar10(nn.Module):
    def __init__(self,output_dim=10,party_num=4,args=None):
        super(BottomModelForCifar10, self).__init__()
        self.btmodel = vgg19(num_classes=output_dim,n_workers=party_num,args=args)

    def forward(self, x):
        x = self.btmodel(x)
        return x

class BottomModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_model(self, half, is_adversary, output_dim=10,n_labels=10,party_num=4,args=None):
        if self.dataset_name in ['CIFAR10','CINIC10L','BHI']:
            return BottomModelForCifar10(output_dim=output_dim,party_num=party_num,args=args)
        elif self.dataset_name == 'Imagenette':
            return BottomModelForImagenette(output_dim=output_dim,party_num=party_num,args=args) 
        else:
            raise Exception('Unknown dataset name!')

    def __call__(self):
        raise NotImplementedError()


class TopModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_model(self,input_dim=10,args=None):
        if self.dataset_name in ['CIFAR10','BHI']:
            return TopModelForCifar10(input_dim,args)
        elif self.dataset_name == 'CINIC10L':
            return TopModelForCifar10(input_dim,args)
        elif self.dataset_name == 'Imagenette':
            return TopModelForCifar10(input_dim,args)
        else:
            raise Exception('Unknown dataset name!')

if __name__ == "__main__":
    demo_model = BottomModel(dataset_name='CIFAR10')
    print(demo_model)
