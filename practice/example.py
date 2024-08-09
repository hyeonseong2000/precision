
# preciosn with tensor/torch 
import numpy as np
import torch
import torch.nn as nn
from   torch.autograd import Function
torch.set_printoptions(precision=20)
@torch.no_grad()
def custom_fp (input, e_width, m_width, required_bin=False):
    if isinstance (input, np.ndarray):
        e_width_exp = np.power(np.array([2.0]), e_width  )                    # 2^exp
        m_width_exp = np.power(np.array([2.0]), m_width  )                    # 2^m
        exp_range   = np.power(np.array([2.0]), e_width  ) / 2                # exp range = 128 , use this because literally exponential width represent exponential range
        max_range   = np.power(np.array([2.0]), exp_range)                    # max range = 2^128 -> 1 1111_1111 0x7
        max_data    = ((np.power(np.array([2.0]), m_width+1) -1) / np.power(np.array([2.0]), m_width)) * max_range 
        # max_data    = (1 + (np.power(np.array([2.0]), m_width) - 1) / np.power(np.array([2.0]), m_width)) * max_range 
        min_data    = np.power(np.array([2.0]), -exp_range+2           )      # because of denormalization, exp 0000...00 is not minimum, 000...001 is minimum
                                                                              # above denorm, it is 2^(-e+2) x 1.0000..00 is minimum, under this value is need denormalized
        denormal_unit_reci= np.power(np.array([2.0]), m_width+exp_range-2 )   # if denorm, smallest unit is 2^(-e+2) x 0.0000...0001
                                                                              #                           = 2^(-e+2) x 2^(-m_width)
                                                                              #             Therefore,   => 2^(e-2) x 2^(m_width) is mult for truncation
                                                                              #                           = 2^(m_width + e - 2)
                                                                              #                           = 2^(126 + 23) x  ( if singel precision )

        def denormal(x, denormal_unit_reci):
            return np.round(x*denormal_unit_reci, 0)/denormal_unit_reci       # use this denormalization minimum unit to truncate smaller than this value.

        output       =  np.clip (input, -max_data, max_data)                  # clipping for representation possible value
        mask         =  (output>0) & (output<min_data)                        # masking denorm value
        output[mask] =   denormal( output[mask], denormal_unit_reci)          # transform denorm value
        mask         =  (output<0) & (output>-min_data)
        output[mask] =  -denormal(-output[mask], denormal_unit_reci)

        zero_mask=(output==0)                                                 # if output equal to zero, it is zero

        # mantissa adjust
        sign   = np.sign(output)  
        output = output*sign                                                  # now, output is absolute value because if sign is -1, mult -1 is 1, and 1 is also 1
        exp    = np.floor(np.log2(output))                                    # extract exponential value
        man    = np.round  (m_width_exp * output * np.power(np.array([2.0]), -exp), 0)  # mult (output x 2^(-e)) with m_width_exp, then output become mantissa precssion
        output = (sign * man) / m_width_exp * np.power(np.array([2.0]),  exp) # then restore mantissa value by man / m_width_exp = 1.xxxx.. and mult with 2^e and sign is become real transformed output
        output [zero_mask] = 0

        if required_bin:            
            man    [zero_mask] = 0                                            # remove nan (x=0, exp=-inf)             
            inf_mask = ~np.isfinite (exp)                                     # if nan, inf inf_mask is true, if 0,1 ... finite then inf_mask is false. ~ is complement operator
                                                                              # if output ~= 0 then, exp can be very high value, so it can be represent inf or NaN it is used instead of very small value like epsilon in log2(output)
            exp_bin  = (exp + exp_range - 1).astype(int)                      # exp_range - 1 => bias, so exp + bias is original exp_bin val
            exp_bin  [inf_mask] = 0                                           # for express inf or Nan ..., exp_bin is become 000...00 
            sign_bin = (sign < 0).astype(int)

            man_bin = (-np.power(np.array([2.0]), m_width) + man).astype(int) # present man is 1100...010 and first 1 represent 2^(m_width)
                                                                              # so, man - 2^(m_width) => 100...010 == 0.100...010 so, it can be a mantissa binary radix directly
            man_bin [man_bin < 0] = 0                                         # 이게 중요하다, 0인 경우 확인, if denorm value, man_bin < 0, is it possible????
            output_bin = sign_bin*2**(e_width+m_width) + exp_bin*2**(m_width) + man_bin

    else: # torch.tensor
        device = "cpu"
        e_width_exp = torch.pow(torch.tensor(2.0, device=device), e_width)
        m_width_exp = torch.pow(torch.tensor(2.0, device=device), m_width)
        exp_range   = torch.pow(torch.tensor(2.0, device=device), e_width) / 2  # exp range = 128
        max_range   = torch.pow(torch.tensor(2.0, device=device), exp_range)    # max range = 2^128 -> 1 1111_1111 0x7
        max_data    = (1 + (torch.pow(torch.tensor(2.0, device=device), m_width) - 1) / torch.pow(torch.tensor(2.0, device=device), m_width)) * max_range
        min_data    = torch.pow(torch.tensor(2.0, device=device), -exp_range+2)
        denormal_unit_reci=torch.pow(torch.tensor(2.0, device=device), m_width+exp_range-2)

        def denormal(x, denormal_unit_reci):
            return torch.round(x*denormal_unit_reci)/denormal_unit_reci

        def log2(x_in):
            return torch.div(torch.log(x_in), torch.log(torch.tensor(2.0, device=device)))
        
        #exp filtering
        output       =    torch.clamp (input, -max_data, max_data)
        mask         =   (output>0) & (output<min_data)
        output[mask] =    denormal( output[mask], denormal_unit_reci)
        mask         =   (output<0) & (output>-min_data)
        output[mask] =   -denormal(-output[mask], denormal_unit_reci)

        zero_mask=(output==0)
        # mantissa adjust
        sign = torch.sign(output)
        output=output*sign
        exp = torch.floor(log2(output))
        man = torch.round(m_width_exp * torch.mul(output, torch.pow(torch.tensor(2.0, device=device), -exp)))
        output = (sign*man)/m_width_exp * torch.pow(torch.tensor(2.0, device=device), exp)
        output[zero_mask]=0

        if required_bin:      
            man [zero_mask] = 0 # remove nan (x=0, exp=-inf)

            man    [zero_mask] = 0 # remove nan (x=0, exp=-inf)             
            #pdb.set_trace ()

            inf_mask = ~torch.isfinite (exp)
            exp_bin  = (exp + exp_range - 1)
            exp_bin  = exp_bin.type (torch.int)
            exp_bin  [inf_mask] = 0
            sign_bin = (sign < 0)
            sign_bin = sign_bin.type (torch.int)

            man_bin = (-torch.pow(torch.tensor(2.0, device=device), m_width) + man)
            man_bin = man_bin.type (torch.int)
            man_bin [man_bin < 0] = 0 # 이게 중요하다, 0인 경우 확인
            output_bin = sign_bin*2**(e_width+m_width) + exp_bin*2**(m_width) + man_bin
        
    if required_bin:
        return output, output_bin
    else:
        return output

@torch.no_grad()
def custom_fxp(input, f_width, i_width, required_bin=False):
    if isinstance (input, np.ndarry):
        i_width_exp = np.power(np.array([2.0]), i_width)    
        f_width_exp = np.power(np.array([2.0]), f_width)
        abs_max     = i_width_exp -np.array([1.0])/f_width_exp            # if i width = 5 , f width = 5, abs_max = 2^5 - 2^(-5) => 2^(i_width) - 2^(-f_width)
        
        # round of rint -> half to even
        output_bin = np.rint (input * f_width_exp, 0)
        output = output_bin / f_width_exp                                 # truncate lower bit then present preicision

        output     = np.clip (output, -abs_max, abs_max)                  # sign bit 
        if required_bin:
            abs_max_bin = abs_max * f_width_exp
            output_bin  = np.clip (output_bin, -abs_max_bin, abs_max_bin)

    else: # torch.tensor
        device="cpu"
        i_width_exp = torch.pow(torch.tensor(2.0, device=device), i_width)
        f_width_exp = torch.pow(torch.tensor(2.0, device=device), f_width)
        abs_max     = i_width_exp - torch.tensor(1.0 , device=device)/ f_width_exp

        output_bin = torch.round(input * f_width_exp, 0)
        output     = output_bin / f_width_exp
        output  = torch.clamp (output, -abs_max, abs_max)
        if required_bin:
            abs_max_bin = abs_max * f_width_exp
            output_bin  = torch.clamp  (output_bin, -abs_max_bin, abs_max_bin)

    if required_bin:
        return output, output_bin
    else:
        return output

def custom_precision (input, e_width, m_width, type="fp", required_bin=False):
    # e_width: exp in fp, frac in fxp
    # m_width: man in fp, int  in fxp
    if type == "fp":
        return  custom_fp  (input, e_width, m_width, required_bin)        
    elif type == "fxp":
        return  custom_fxp (input, e_width, m_width, required_bin)
    else:
        assert False, f"type: {type} not supported in custom_precision"

# revised: required binary for debugging
# not support quantize gradient (only inference)
class Convert_Precision_Function (Function):
    @staticmethod
    def forward (ctx, input, e_width, m_width, type="fp", required_bin=False):
        ctx.constant = (e_width, m_width, type, required_bin)
        out = custom_precision (input, e_width, m_width, type, required_bin)
        return out
    @staticmethod
    def backward (ctx, grad):
        out = grad
        return out, None, None, None, None, None, None, None

class Convert_Precision (nn.Module):
    def __init__ (self, e_width, m_width, type="fp"):
        super (Convert_Precision, self).__init__()
        self.e_width = e_width
        self.m_width = m_width
        self.type    = type

    def forward (self, input, required_bin=False):
        out = Convert_Precision_Function.apply (input, self.e_width, self.m_width, self.type, required_bin)
        return out
    


a = torch.tensor([131009e+50, 0.0000000059604644775390625 ,0.141411124241])
model = Convert_Precision(5, 10, "fp")

b,c = model(a , True)
print(b, "\n", c)



