
"""
训练需要啥
1.dataset:img,label
2. 网络 :out
3.损失函数：conf（bce），cxcywh（mse）,cls(cross)
4.优化器
"""



from dataset import My_Dataset
from torch.utils.data import DataLoader
from  yolov3 import My_Yolov3
from torch import nn
import torch


class Train:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = My_Dataset(r"D:\PycharmProjects\20260717\little_data\Parse_label.txt")
        self.dataloder = DataLoader(self.dataset,batch_size=1,shuffle=True)
        self.net = My_Yolov3().to(self.device)

        self.conf_loss = nn.BCEWithLogitsLoss()
        self.obj_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.net.parameters())
    def loss_fn(self,out,target,scale =0.9):
        #label_13:torch.Size([2, 13, 13, 3, 9])
        #out_13:torch.Size([2, 27, 13, 13])
        #转换模型输出的形状torch.Size([2, 13, 13, 27])
        out = out.permute(0,2,3,1)
        #拆分模型输出的形状 #out_13:torch.Size([2, 13, 13, 3, 9])
        out = torch.reshape(out,(out.shape[0],out.shape[1],out.shape[2],3,-1))


        #正样本
        #满足条件的正样本
        target_mask_13 = target[...,0]>0
        target_obj =target[target_mask_13]
        #取出满足条件的预测样本
        out_obj  = out[target_mask_13]

        conf_loss = self.conf_loss(out_obj[:,0],target_obj[:,0])
        obj_loss = self.obj_loss(out_obj[:,1:5],target_obj[:,1:5])
        cls_loss = self.cls_loss(out_obj[:,5:],torch.argmax(target_obj[:,5:],dim = 1))

        true_loss = conf_loss+obj_loss+cls_loss

        #负样本
        target_noobj_mask = target[...,0]==0
        target_noobj = target[target_noobj_mask]
        out_noobj = target[target_noobj_mask]

        conf_no_loss = self.conf_loss(out_noobj,target_noobj)

        total_loss = true_loss*scale+conf_no_loss*(1-scale)
        return total_loss
    def train(self,):
        self.net.train()
        for epoch in range(700):
            for batch,(label_13,label_26,label_52,img_tensor) in enumerate(self.dataloder):
                label_13,label_26,label_52 ,img_tensor= label_13.to(self.device),label_26.to(self.device),label_52.to(self.device),img_tensor.to(self.device)
                out_13,out_26,out_52 = self.net(img_tensor)

                loss13 = self.loss_fn(out_13,label_13)
                loss26 = self.loss_fn(out_26,label_26)
                loss52 = self.loss_fn(out_52,label_52)

                loss = loss13+loss26+loss52
                print(f"epoch==={epoch},j======{epoch}, loss======{loss.item()}")

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            if epoch and epoch%100==0 :
                torch.save(self.net.state_dict(),f"best_loss_{epoch}.pt")
                print("loss save success")



if __name__=="__main__":
    trainer = Train()
    trainer.train()


