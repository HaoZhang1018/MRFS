import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat([Y, Cr, Cb], dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def Sobelxy(x):
    kernelx = [[-1, 0, 1],
              [-2,0 , 2],
              [-1, 0, 1]]
    kernely = [[1, 2, 1],
              [0,0 , 0],
              [-1, -2, -1]]
    kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
    kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
    weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
    weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    
    sobelx=F.conv2d(x, weightx, padding=1)
    sobely=F.conv2d(x, weighty, padding=1)
    return sobelx, sobely


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.l1_loss =  nn.L1Loss()

    def forward(self, input_xy, output, Mask):
        input_vis, input_ir = input_xy 

        Fuse = output * Mask
        YCbCr_Fuse = RGB2YCrCb(Fuse) 
        Y_Fuse  = YCbCr_Fuse[:,0:1,:,:]
        Cr_Fuse = YCbCr_Fuse[:,1:2,:,:]
        Cb_Fuse = YCbCr_Fuse[:,2:,:,:]  
        

        R_vis = torchvision.transforms.functional.adjust_gamma(input_vis, 0.5, 1)
        YCbCr_R_vis = RGB2YCrCb(R_vis) 
        Y_R_vis = YCbCr_R_vis[:,0:1,:,:]
        Cr_R_vis = YCbCr_R_vis[:,1:2,:,:]
        Cb_R_vis = YCbCr_R_vis[:,2:,:,:]          
        
                        
        R_ir = torchvision.transforms.functional.adjust_contrast(input_ir, 1.8)


        Fuse_R = torch.unsqueeze(Fuse[:,0,:,:],1)
        Fuse_G = torch.unsqueeze(Fuse[:,1,:,:],1)
        Fuse_B = torch.unsqueeze(Fuse[:,2,:,:],1)
        Fuse_R_grad_x,Fuse_R_grad_y =   Sobelxy(Fuse_R)
        Fuse_G_grad_x,Fuse_G_grad_y =   Sobelxy(Fuse_G)
        Fuse_B_grad_x,Fuse_B_grad_y =   Sobelxy(Fuse_B)
        Fuse_grad_x = torch.cat([Fuse_R_grad_x, Fuse_G_grad_x, Fuse_B_grad_x], 1)
        Fuse_grad_y = torch.cat([Fuse_R_grad_y, Fuse_G_grad_y, Fuse_B_grad_y], 1)


        R_VIS_R = torch.unsqueeze(R_vis[:,0,:,:],1)
        R_VIS_G = torch.unsqueeze(R_vis[:,1,:,:],1)
        R_VIS_B = torch.unsqueeze(R_vis[:,2,:,:],1)
        R_VIS_R_grad_x, R_VIS_R_grad_y =   Sobelxy(R_VIS_R)
        R_VIS_G_grad_x, R_VIS_G_grad_y =   Sobelxy(R_VIS_G)
        R_VIS_B_grad_x, R_VIS_B_grad_y =   Sobelxy(R_VIS_B)
        R_VIS_grad_x = torch.cat([R_VIS_R_grad_x, R_VIS_G_grad_x, R_VIS_B_grad_x], 1)
        R_VIS_grad_y = torch.cat([R_VIS_R_grad_y, R_VIS_G_grad_y, R_VIS_B_grad_y], 1)


        R_IR_R = torch.unsqueeze(R_ir[:,0,:,:],1)
        R_IR_G = torch.unsqueeze(R_ir[:,1,:,:],1)
        R_IR_B = torch.unsqueeze(R_ir[:,2,:,:],1)
        R_IR_R_grad_x,R_IR_R_grad_y =   Sobelxy(R_IR_R)
        R_IR_G_grad_x,R_IR_G_grad_y =   Sobelxy(R_IR_G)
        R_IR_B_grad_x,R_IR_B_grad_y =   Sobelxy(R_IR_B)
        R_IR_grad_x = torch.cat([R_IR_R_grad_x, R_IR_G_grad_x,R_IR_B_grad_x], 1)
        R_IR_grad_y = torch.cat([R_IR_R_grad_y, R_IR_G_grad_y,R_IR_B_grad_y], 1)


        joint_grad_x = torch.maximum(R_VIS_grad_x, R_IR_grad_x)
        joint_grad_y = torch.maximum(R_VIS_grad_y, R_IR_grad_y)
        joint_int  = torch.maximum(R_vis, R_ir)
        
        
        con_loss = self.l1_loss(Fuse, joint_int)
        gradient_loss = 0.5 * self.l1_loss(Fuse_grad_x, joint_grad_x) + 0.5 * self.l1_loss(Fuse_grad_y, joint_grad_y)
        color_loss = self.l1_loss(Cb_Fuse, Cb_R_vis) + self.l1_loss(Cr_Fuse, Cr_R_vis)


        fusion_loss_total = 0.5 * con_loss  + 0.2 * gradient_loss  + 1 * color_loss

        return fusion_loss_total

class MakeLoss(nn.Module):
    def __init__(self, background):
        super(MakeLoss, self).__init__()

        self.semantic_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=background)
        self.FusionLoss = FusionLoss()

    def forward(self, inputs, outputs, Mask, label):
        input_vis, input_ir = inputs
        out_semantic, Fus_img = outputs
        fusion_loss_total = self.FusionLoss(inputs, Fus_img, Mask)

        semantic_loss_total = self.semantic_loss(out_semantic,label)

        loss = 1 * semantic_loss_total + 0.1 * fusion_loss_total

        return loss, semantic_loss_total, fusion_loss_total
