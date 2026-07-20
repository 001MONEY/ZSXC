

"""
训练需要啥
1.dataset:img,label
2. 网络 :out
3.损失函数：conf（bce），cxcywh（mse）,cls(cross)
4.优化器
"""


from dataset import My_Dataset
from yolov3 import My_Yolov3
import torch
from torch.utils.data import DataLoader
from torch import nn


class Train:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_dataloder = DataLoader(My_Dataset(r"D:\PycharmProjects\20260717\little_data\Parse_label.txt"),batch_size=5,shuffle=True)
        self.net = My_Yolov3().to(self.device)
        # self.pt_path = r"E:\zs_kejian\AIcourse\07-YOLOV3\best_loss.pt"
        # self.net.load_state_dict(torch.load(self.pt_path,map_location=self.device))
        #损失及优化器
        self.conf_loss = nn.BCEWithLogitsLoss()
        self.obj_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.net.parameters())

    
    def loss_fn(self,pred,target,factor=0.9):
        # pred:n 3*(5+4) h w
        #label :13 13  3 5+4 
        pred = pred.permute(0,2,3,1)  #n h w 3*(5+4)
        pred = torch.reshape(pred,(pred.size(0),pred.size(1),pred.size(2),3,-1))
        #pred：torch.Size([10, 13, 13, 3, 9])
        #target：torch.Size([10, 13, 13, 3, 9])

        
        #正样本
        target_mask = target[...,0]>0
        #取出满足条件的标签数据N9，torch.Size([84, 9])
        target_obj = target[target_mask]
        #取出满足条件的预测数据N9，torch.Size([84, 9])
        pred_obj = pred[target_mask]
        
        conf_loss = self.conf_loss(pred_obj[:,0],target_obj[:,0])
  

        obj_loss  =self.obj_loss(pred_obj[:,1:5],target_obj[:,1:5])
        cls_loss = self.cls_loss(pred_obj[:,5:],torch.argmax(target_obj[:,5:],dim=1))
        true_loss = conf_loss+obj_loss+cls_loss
        
        #负样本
        target_noobj_mask = target[...,0]==0
        target_noobj = target[target_noobj_mask]
        pred_noobj = pred[target_noobj_mask]

        conf_loss_noobj = self.conf_loss(pred_noobj[:,0],target_noobj[:,0])
        total_loss_obj = true_loss*factor+(1-factor)*conf_loss_noobj
        #正负样本严重失衡 → 给正样本 90% 权重、负样本 10% 权重，防止模型被背景淹没。
        return total_loss_obj


    def train(self):
        self.net.train()
        for epoch in range(700):
            for j ,(label_13,label_26,label_52 ,img_data)in enumerate(self.train_dataloder):
                #13 13  3 5+4 
                #26 26 3 5+4 
                #52 52 3 5+4
                label_13,label_26,label_52 = label_13.to(self.device),label_26.to(self.device),label_52.to(self.device)
                img_data = img_data.to(self.device)
                # n 3*(5+4) h w
                out_13,out_26,out_52 = self.net(img_data)
                #损失：cof, cx cy ,w h ,*cls
                loss13 = self.loss_fn(out_13,label_13)
                loss26 = self.loss_fn(out_26,label_26)
                loss52 = self.loss_fn(out_52,label_52)

                loss = loss13+loss26+loss52
                print(f"epoch==={epoch},j======{j}, loss======{loss.item()}")

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            if epoch and epoch%100==0 :
                torch.save(self.net.state_dict(),f"best_loss_{epoch}00.pt")
                print("loss save success")


        
if __name__=="__main__":
    trainer =Train()
    trainer.train()

