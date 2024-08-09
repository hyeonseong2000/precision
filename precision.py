import torch
from torch.autograd import Function


# Floating point to Fixed point
def fxp_quant(input, n_int, n_frac, mode="trunc", device = "cpu"):
    # n_int includes Sign Bit (2's complement)
    max_val = (2** (n_int)) - (2**(-n_frac))
    min_val = -max_val
    sf = 2 ** n_frac #scaling factor = 2**(n_frac)

    assert mode in ["trunc", "round", "stochastic"] , "Quantize Mode Must be 'trunc' or 'round' or 'stochastic'"
    
    input = input.to(device)
    #abs_input = torch.abs(input)
    #max_input = torch.max(abs_input)
    #norm_input = input / max_input # Normalized input
    #input = norm_input * (2 ** (n_int - 1)) # Scaling Norm. input, if n_int = 5 -> -16 < input < 16
    
    # Restrict the number with given bit precision (Fractional Width)
    # Quantization Rules
    if(mode == "trunc"):
        input_trunc = torch.floor(input * sf)/sf # Truncate Fractional Bit
    elif(mode == "round"):
        input_trunc = torch.round(input * sf)/sf  # Round to Nearest
    elif(mode == "stochastic"):
        decimal_part = input * sf - torch.floor(input * sf)
        rdn = torch.where(torch.rand_like(decimal_part) < decimal_part , 0 , 1)
        input_trunc = torch.floor(input * sf + rdn)/sf

    
    # Saturate Overflow Vals
    clipped_input = torch.clamp(input_trunc, min_val, max_val)

    return clipped_input

# Floating point to Floating point
def fp_quant(input, n_exp, n_man, mode="trunc", device = "cpu"):
    bias = (2 ** (n_exp - 1)) -1
    exp_min = (2** ((-bias) + 1))
    #exp_max = (2** (bias + 1))
    exp_max = torch.pow(torch.tensor(2.0, device = device) , (bias + 1))
    #man_max = 2 - (2**(-n_man))
    man_max = (1 + (2**n_man -1 )/2**n_man)
    man_min = (2 ** (-n_man))
    min_val = exp_min * man_min
    max_val = exp_max * man_max
    epsilon = 1e-16
    man_sf = 2 ** n_man  # Mantissa Scaling factor


    # Again, Check Overflow
    input_clipped = torch.clamp(input, min = -max_val, max = max_val)
    
    mask = (input_clipped>0) & (input_clipped < min_val)
    input_clipped [mask] = 0
    mask = (input_clipped<0) & (input_clipped > -min_val)
    input_clipped [mask] = 0

    zero_mask = (input_clipped == 0)

    input_abs = torch.abs(input_clipped)


    assert mode in ["trunc", "round", "stochastic"] , "Quantize Mode Must be 'trunc' or 'round' or 'stochastic'"
    


    # Extract exp value
    input_exp = torch.log2(input_abs).to(device)
    input_exp = torch.floor(input_exp).to(device)


    # For Denenormalize
    input_exp = torch.where(input_exp <torch.tensor((-bias)+1).float().to(device), torch.tensor((-bias)+1).float().to(device), input_exp).to(device) 
    
    
       
    
    # When input_exp < (-bias + 1), Denorm, and input_man < 1 (Denormalized number!)
    input_man = input_abs / (2 ** input_exp)  # If Denorm, input_man = 0.xxxx , else input_man == 1.xxxx

    # Same with Fixed point, Restrict the number with given bit precision (Mantissa Width)
    # Mantissa Quantization
    if(mode == "trunc"):
        man_trunc = torch.floor(input_man * man_sf)/man_sf # Truncate Fractional Bit
    elif(mode == "round"):
        man_trunc = torch.round(input_man * man_sf)/man_sf  # Round to Nearest
    elif(mode == "stochastic"):
        #decimal_part = input_man * man_sf - torch.floor(input_man * man_sf)
        #rdn = torch.where(torch.rand_like(decimal_part) < decimal_part , 0 , 1)
        rdn = torch.rand_like(input_man)
        man_trunc = torch.floor(input_man * man_sf + rdn)/man_sf
 

    # Value restore ( mantissa * 2^(exp)) if Denorm case, mantissa = 0.xxxx, exp = (-bias + 1)
    input_quantized = man_trunc * (2 ** input_exp)

    

    # Attach Sign Bit
    signed_input = input_quantized * torch.sign(input)

    return signed_input

def quantization(input, type="fxp", n_exp = 5 , n_man = 10, mode = "trunc", device = "cpu" ):
    if (type == "fxp") :
        input = input.to(device)
        quant_output = fxp_quant(input, n_man, n_exp, mode , device)
    elif (type == "fp"):
        input = input.to(device)
        quant_output = fp_quant(input, n_exp, n_man, mode, device)

    return quant_output


class Quant(Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, type="fxp", n_exp =5 , n_man = 10, mode = "trunc", device = "cpu"):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.type = type
        ctx.n_exp = n_exp
        ctx.n_man = n_man
        ctx.mode = mode
        ctx.device = device
        input = input.to(device)
        output = quantization(input, type, n_exp, n_man, mode, device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_output = grad_output.to(ctx.device)
        #print(torch.min(torch.abs(grad_output)[torch.abs(grad_output) >0]))
        grad_input = grad_output
        
        
        return grad_input ,None, None, None, None, None


class PrecisionLayer(torch.nn.Module):
    def __init__(self, type = "fxp", n_exp = 5, n_man = 10, mode = "trunc", device = "cpu" ):
        super(PrecisionLayer,self).__init__()
        self.type = type
        self.n_exp = n_exp
        self.n_man = n_man
        self.mode = mode
        self.device = device

    
    def forward(self, input):
        input = input.to(self.device)
        return Quant.apply(input, self.type, self.n_exp, self.n_man, self.mode, self.device)
    



    def extra_repr(self):
        # (optional) Set the extra information about this module.
        # You can test it by printing an object of this class
        return f'type={self.type}, n_exp={self.n_exp}, n_man={self.n_man}, mode={self.mode}, device={self.device}'


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        torch.nn.init.xavier_uniform_(self.fc.weight)


    def forward(self, x, type = "fxp", n_exp = 5, n_man = 10, mode = "trunc", device = "cpu" ):
        x = x.to(device)
        out = Quant.apply(x , type , n_exp, n_man, mode, device)
        out = self.layer1(x)
        out = Quant.apply(out , type , n_exp, n_man, mode, device)
        out = self.layer2(out)
        out = Quant.apply(out , type , n_exp, n_man, mode, device)
        out = out.view(out.size(0), -1)   
        out = self.fc(out)
        return out



class Alexnet(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 96, kernel_size=11, stride=4), # conv1
            torch.nn.ReLU(inplace=True),
            torch.nn.LocalResponseNorm(2),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2), # conv2
            torch.nn.ReLU(inplace=True),
            torch.nn.LocalResponseNorm(2),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), # conv3
            torch.nn.ReLU(inplace=True)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), # conv4
            torch.nn.ReLU(inplace=True)
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), # conv5
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((6,6))				

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(6*6*256, 4096), # fc1
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096), # fc2
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, num_classes) # fc3
        )
        
    def forward(self, x, type = "fxp", n_exp = 5, n_man = 10, mode = "trunc", device = "cpu"):
        x = Quant.apply(x , type , n_exp, n_man, mode, device)
        x = self.layer1(x)
        x = Quant.apply(x , type , n_exp, n_man, mode, device)
        x = self.layer2(x)
        x = Quant.apply(x , type , n_exp, n_man, mode, device)
        x = self.layer3(x)
        x = Quant.apply(x , type , n_exp, n_man, mode, device)
        x = self.layer4(x)
        x = Quant.apply(x , type , n_exp, n_man, mode, device)
        x = self.layer5(x)
        x = Quant.apply(x , type , n_exp, n_man, mode, device)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = Quant.apply(x , type , n_exp, n_man, mode, device)
        x = self.classifier(x)
        x = Quant.apply(x , type , n_exp, n_man, mode, device)
        return x





if __name__ == '__main__':
    # debugging codes here
    import torch
    import torchvision.datasets as dsets
    import torchvision.transforms as transforms
    import torch.nn.init
    import argparse
    import os

    parser = argparse.ArgumentParser()

    ############ wh revised ##############
    # for argument parser
    args_from_cmd = parser.parse_known_args()[1]
    print(args_from_cmd)
    args_list= []
    for arg in args_from_cmd:
        if '--' in arg: args_list.append(arg[2:])
    ######################################
    
    parser.add_argument('--config', type=str, default="/lib/configs/basic.yaml", help="path for config file w/ yaml format")
    parser.add_argument('--workspace', type=str, default='workspace')
    ### precision quantization options
    parser.add_argument('--dtype',      type=str    , default='fxp' ,             help='Choose Data Type to Quantize: "fxp" or "fp"')
    parser.add_argument('--exp',          type=int    , default=5    , help='Exponent/Integer Bit-width') 
    parser.add_argument('--mant',          type=int    , default = 10 , help = 'Mantissa/Fractional Bit-width')
    parser.add_argument('--mode', type=str    , default = 'trunc' , help = "Quantization Rule: 'trunc' or 'round' or 'stochastic")

    
    
    
    ### training options
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=100, help="initial training batch size")
    parser.add_argument('--epoch', type=int, default=20, help="initial epoch")
    parser.add_argument('--train', action='store_true', help="training mode")
    parser.add_argument('--resume_from', type=str , default=None, help="Path to resume training model")

    ### dataset options
    parser.add_argument('--data_format', type=str, default='mnist', choices=['mnist', 'cifar10', 'imagenet'] ,help="choose dataset")


    ### testing options
    parser.add_argument('--test', action='store_true', help="test mode")


    opt = parser.parse_args()

    ########## wh revised ##########
    # get configs from config file 
    import lib.config as config    
    cfg  =  config.parse_config_yaml_dict(yaml_path=opt.config)
    opt  =  config.convert_to_namespace  (dict_in=cfg, args=opt, dont_care=args_list)


    # log opt as yaml in log path 
    import yaml
    config_dir = os.path.join(opt.workspace, 'config')
    os.makedirs (config_dir, exist_ok=True)
    config_path = os.path.join (config_dir, 'config.yaml')
    yaml.dump(opt, open (config_path, 'w'))

    ################################ 
    



    print("--------------------")

    if(opt.dtype == "fxp"):
        print("Type         : ", opt.dtype)
        print("Integer Bit  : ", opt.exp)
        print("Fraction Bit : ", opt.mant)
    elif(opt.dtype == "fp"):
        print("Type         : ", opt.dtype)
        print("Exponent Bit : ", opt.exp)
        print("Mantissa Bit : ", opt.mant)

    print("Data Set     : ", opt.data_format)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Present Device : ", device)
    print("Learning Rate : ", opt.lr)
    print("Training Batch size : ", opt.batch_size)
    print("Rounding Mode : ", opt.mode)

    if(opt.resume_from):
        print("Resume Trainig...")

    
    # torch.manual_seed(777)

   
    # if device == 'cuda':
    #     torch.cuda.manual_seed_all(777)

    batch_size = opt.batch_size
    
    assert opt.dtype in ["fxp", "fp"], "Data Type Must be 'fp' or 'fxp'"
    assert opt.data_format in ["mnist", "cifar10" , 'imagenet'], "Data Format Must be 'mnist' or 'cifar10' or 'imagenet'"
    

    if (opt.data_format == "mnist"):

        # Model declaration
        data_shape = [1,28,28]
        model = CNN().to(device)



        train_dataset = dsets.MNIST(root='/home/sslunder2/dataset/MNIST_data/', 
                                  train=True, 
                                  transform=transforms.ToTensor(), 
                                  download=True)
       
        test_dataset = dsets.MNIST(root='/home/sslunder2/dataset/MNIST_data/', 
                                train=False, 
                                transform=transforms.ToTensor(),
                                download=True)


    elif (opt.data_format == "cifar10"):
        
        # Model declaration
        data_shape = [3,227,227]
        model = Alexnet(3,10).to(device)

        transform = transforms.Compose([
                                        transforms.Resize(227),
                                        transforms.CenterCrop(227),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                        ])

        train_dataset = dsets.CIFAR10(root='/home/sslunder2/dataset/CIFAR10_data/', 
                                      train=True, 
                                      transform=transform, 
                                      download=True)

        test_dataset = dsets.CIFAR10(root='/home/sslunder2/dataset/CIFAR10_data/', 
                                train=False, 
                                transform=transform,
                                download=True)
        

        
    elif (opt.data_format == "imagenet"):

        # Model declaration
        data_shape = [3,227,227]
        model = Alexnet(3,1000).to(device)

        transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        train_dataset = dsets.ImageNet(root='/home/sslunder2/dataset/ImageNet_data/', 
                                      train=True, 
                                      transform=transforms.ToTensor(), 
                                      download=True)

        test_dataset = dsets.ImageNet(root='/home/sslunder2/dataset/ImageNet_data/', 
                                        train=False, 
                                        transform=transforms.ToTensor(),
                                        download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=True)




    learning_rate = opt.lr
    training_epochs = opt.epoch

    criterion = torch.nn.CrossEntropyLoss().to(device)    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_batch = len(train_loader)
    

    if (opt.resume_from):
        checkpoint = torch.load(opt.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        start_epoch = epoch + 1
    else:
        start_epoch = 0



    # Training for Quantization Network
    if (opt.train) :
        model.train()
        min_cost =1000
        if(opt.resume_from):
            min_cost = checkpoint['min_cost']
        for epoch in range(start_epoch, training_epochs):
            avg_cost = 0
            
            for X, Y in train_loader: 
                # image is already size of (28x28), no reshape
                # label is not one-hot encoded
                X = X.to(device)
                Y = Y.to(device)

                optimizer.zero_grad()
                hypothesis = model(X , opt.dtype , opt.exp , opt.mant , opt.mode , device).to(device)
                cost = criterion(hypothesis, Y)
                cost.backward()
                optimizer.step()

                avg_cost += cost / total_batch
                
            
            if (min_cost > avg_cost):
                min_cost = avg_cost
                torch.save({
                            'epoch':epoch,
                            'optimizer_state_dict':optimizer.state_dict(),
                            'model_state_dict':model.state_dict(),
                            'min_cost':min_cost
                            }
                            , f'./saved_model/saved_model.pt')
            
            print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

    


    


    if (opt.test) :
        model_data = torch.load("./saved_model/saved_model.pt", map_location=device)

        model.load_state_dict(model_data['model_state_dict'])

        # eval torch.no_grad()  
        with torch.no_grad():
            model.eval()
            corr = 0
            running_loss = 0
            
            for img, lbl in test_loader:
                # Upload data to the device.
                img, lbl = img.to(device), lbl.to(device)
                
                # Results are derived by performing forward propagation on the model.
                output = model(img, opt.dtype , opt.exp , opt.mant , opt.mode , device).to(device)
                
            
                _, pred = output.max(dim=1)
                
            
                corr += torch.sum(pred.eq(lbl)).item()
                
                
                running_loss += criterion(output, lbl).item() * img.size(0)
        
            
            acc = float(corr) / len(test_loader.dataset)



            print(f'Accuracy: {100 * acc : .3f} %' )

    print("--------------------")
