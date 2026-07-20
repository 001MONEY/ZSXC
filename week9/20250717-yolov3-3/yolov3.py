
from torch import nn
import torch




class CBL(nn.Module):
    def __init__(self,c_in,c_out,k, s):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(c_in,c_out,k,s,padding=k//2,bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU()

        )
    def forward(self,x):
        return self.mod(x)
    

class Residual(nn.Module):
    def __init__(self, c_in,):
        super().__init__()
        self.mod = nn.Sequential(
            CBL(c_in,c_in//2,1,1),
            CBL(c_in//2,c_in,3,1),

        )
    def forward(self,x):
        return self.mod(x)+x
        


class My_darknet53(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.sub_model_52 =nn.Sequential(
            CBL(3,32,3,1),
            CBL(32,64,3,2),
            Residual(64),
            CBL(64,128,3,2),
            Residual(128),
            Residual(128),
            CBL(128,256,3,2),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
        )

        self.sub_model_26= nn.Sequential(
            CBL(256,512,3,2),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
        )

        self.sub_model_13= nn.Sequential(
            CBL(512,1024,3,2),
            Residual(1024),
            Residual(1024),
            Residual(1024),
            Residual(1024),

        )
    def forward(self,x):
        h_52 = self.sub_model_52(x)
        h_26 = self.sub_model_26(h_52)
        h_13 = self.sub_model_13(h_26)
        return h_52,h_26,h_13



class My_darknet53_v2(nn.Module):
    def __init__(self,):
        super().__init__()
        self.input_layer = nn.Sequential(
            CBL(3,32,3,1)
        )
        self.block1 = nn.Sequential(
            CBL(32,64,3,2),
            Residual(64),
        )
        self.block2= nn.Sequential(
            CBL(64,128,3,2),
            Residual(128),
            Residual(128),
        )
        self.block3= nn.Sequential(
            CBL(128,256,3,2),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
        )
        self.block4= nn.Sequential(
            CBL(256,512,3,2),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
        )

        self.block5= nn.Sequential(
            CBL(512,1024,3,2),
            Residual(1024),
            Residual(1024),
            Residual(1024),
            Residual(1024),
        )

    def forward(self,x):
        out_layer = self.input_layer(x)
        block1_out = self.block1(out_layer)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)
        block4_out = self.block4(block3_out)
        block5_out = self.block5(block4_out)
        return  block3_out,block4_out,block5_out



class My_darknet53_v3(nn.Module):
    def __init__(self,channels = [32,64,128,256,512,1024],block_nums = [1,2,8,8,4]):
        super().__init__()
        self.input_layer = nn.Sequential(
            CBL(3,32,3,1)
        )
        layers = []
        for idx,block_num in enumerate(block_nums):
            channle_in ,channle_out=channels[idx] ,channels[idx+1]
            layer = self.make_layers(channle_in,channle_out,block_num)
            layers.append(layer)
        self.stages = nn.Sequential(*layers)


    def make_layers(self,c_in,c_out,block_nums):
        layers = [CBL(c_in,c_out,3,2)]
        for _ in range(block_nums):
            layers.append(Residual(c_out))
        return nn.Sequential(*layers)


    def forward(self,x):
        x = self.input_layer(x)
        out_52 = self.stages[:3](x)
        out_26 = self.stages[3](out_52)
        out_13 = self.stages[4](out_26)

       
        return  out_52,out_26,out_13


class CBLSET(nn.Module):
    def __init__(self,c_in,c_out ):
        super().__init__()
        self.sub_mod = nn.Sequential(
            CBL(c_in,c_out,1,1),
            CBL(c_out,c_in,3,1),
            CBL(c_in,c_out,1,1),
            CBL(c_out,c_in,3,1),
            CBL(c_in,c_out,1,1),
        )
    def forward(self,x):
        return self.sub_mod(x)


class Up_sample(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.sub_mod = nn.Upsample(scale_factor=2,mode= "nearest")
        
    def forward(self,x):
        return self.sub_mod(x)


CLS_NUMS = 4
class My_Yolov3(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bone = My_darknet53_v3()
        self.cov_set_13 = CBLSET(1024,512)
        #neck_out_13
        self.out_neck_13 = nn.Sequential(
            CBL(512,1024,3,1),
             nn.Conv2d(1024,3*(5+CLS_NUMS),1,1)
        )

        #13—->26上采样
        self.up_13=nn.Sequential(
            CBL(512,256,1,1),
            Up_sample()
        )

        self.cov_set_26 = CBLSET(512+256,256)
        #neck_out_26
        self.out_neck_26 = nn.Sequential(
            CBL(256,512,3,1),
            nn.Conv2d(512,3*(5+CLS_NUMS),1,1) #TODO
        )

        #26—->52上采样
        self.up_26_52=nn.Sequential(
            CBL(256,128,1,1),
            Up_sample()
        )
        #neck_out_52

        self.neck_52 =nn.Sequential(
            CBLSET(128+256,128),
            CBL(128,256,3,1),
            nn.Conv2d(256,3*(5+CLS_NUMS),1,1)
        )
    def forward(self,x):
        out_52,out_26,out_13  =self.bone(x)
        cov_set_13_out = self.cov_set_13(out_13)
        
        out_neck_13 = self.out_neck_13(cov_set_13_out)

        up_13_out_26 = self.up_13(cov_set_13_out)
        contate_26 = torch.cat((up_13_out_26,out_26),dim = 1)  #torch.Size([1, 768, 26, 26])
        cov_set_26_out = self.cov_set_26(contate_26)

        out_neck_26 = self.out_neck_26(cov_set_26_out)
        up_26_out_52 =self.up_26_52(cov_set_26_out)
        
        concate_52 =torch.cat((up_26_out_52,out_52),dim = 1)
        out_neck_52 = self.neck_52(concate_52)
        return out_neck_13,out_neck_26,out_neck_52


    



if __name__=="__main__":
    x = torch.randn(1,3,416,416)
    # net = CBLSET(1024,512)
    # net = Up_sample()
    net = My_Yolov3()
    # net(x)
    # print(net(x).shape)
    # x= torch.randn(1,64,208,280)
    # net  =Residual(64)
    
    # net =My_darknet53_v3()
    out = net(x)[0][:,1,:,:]
    print(out)
    print()
    # print(net(x)[0].shape)
    # print(net(x)[1].shape)
    # print(net(x)[2].shape)