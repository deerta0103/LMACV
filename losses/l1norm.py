import torch
import torch.nn as nn
import torch.nn.functional as F


class L1normloss(nn.Module):
    def __init__(self):
        super(L1normloss, self).__init__()

    def forward(self,image_vi,image_ir,generate_img):
        # image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_vi,image_ir)
        loss_in=F.l1_loss(image_ir,generate_img)
        loss_in=loss_in+F.l1_loss(image_vi,generate_img)
        return loss_in
