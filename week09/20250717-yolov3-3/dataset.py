


import torch 
import cv2
import numpy as np 
from torch.utils.data import Dataset
import os
from torchvision import transforms
import cfg
import math
from PIL import Image
# DATAPATH = r"datas\YOLODataset_origin_labels"
DATAPATH=r"D:\PycharmProjects\20260717\little_data"
tr_tranform = transforms.Compose([transforms.ToTensor()])  #单目标检测时候datase图像的归一化是全部一步一步写得  

class My_Dataset(Dataset):
    def __init__(self,root_path:str):
        super().__init__()
        self.dataset = open(root_path,"r",encoding="utf-8").readlines()
      
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        line = self.dataset[index]
       
        line = line.split()
        # print(line)
        img_path = os.path.join(DATAPATH,line[0])
        #输入处理图像
        # img_data = cv2.imread(img_path)
        # img_tensor = tr_tranform(cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB))
        img_data= Image.open(img_path).convert("RGB")
        img_tensor = tr_tranform(img_data)
        
        labels =np.array([float(i) for i in line[1:]])
        labels =np.split(labels,len(labels)//5)
        label = {}
        for feature_size,ANCHAOR_GROUPS in cfg.ANCHORS_GROUPS.items():
            label[feature_size] = torch.zeros((feature_size,feature_size,3,5+cfg.CLASS_NUM))# label{"13":13,13,3,9}
            for bbox in labels:
                cls_,cx,cy,gt_w,gt_h = torch.tensor(bbox,dtype=torch.float32)
                #中心点偏移量 x1/感受视野  整数，小数
                recep_fild = cfg.IMG_SIZE/feature_size   #32
                offset_cx,cx_idx = math.modf(cx/recep_fild)  #offset_cx=0.5625,cx_idx = 6.0
                offset_cy,cy_idx = math.modf(cy/recep_fild)  # offxy = 0.6093 cy_idx = 6.0
                #w,h 
                for idx,ANCHAOR_GROUP in enumerate(ANCHAOR_GROUPS):
                    anchor_w,anchor_h= ANCHAOR_GROUP
                    offset_w = torch.log(gt_w/anchor_w)
                    offset_h = torch.log(gt_h/anchor_h)
                    conf = 1
                    
                    #onehot
                    cls_onehot = torch.nn.functional.one_hot(torch.tensor(int(cls_)),len(cfg.classes))
                    #conf,offset_cx,offset_cy,offset_w,offset_h,*cls_onehot
                    label[feature_size][int(cy_idx),int(cx_idx),idx] =torch.tensor([conf,offset_cx,offset_cy,offset_w,offset_h,*cls_onehot],dtype=torch.float32)
        return label[13],label[26],label[52],img_tensor
                




if __name__=="__main__":
    dataset = My_Dataset(r"D:\PycharmProjects\20260717\little_data\Parse_label.txt")
    label13,label26,label52,img = dataset[0]
    # label13 ：13 13 3 9
    label_13_1 = label13[:,:,0]
    print(label_13_1.shape)   #torch.Size([13, 13, 9])
    label13_1_mask = label_13_1[:,:,0]>0
    print(label13_1_mask)     #torch.Size([13, 13])
    print(label13_1_mask.shape)  ##torch.Size([13, 13])
    print( label_13_1[label13_1_mask])



    # print(label13)



   