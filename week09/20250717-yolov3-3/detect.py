
import torch
from torchvision import transforms
from yolov3 import My_Yolov3
import os
from PIL import Image,ImageDraw
import cfg
import numpy as np
import cv2
torch.set_printoptions(sci_mode=False)
tr_tranform = transforms.Compose([transforms.ToTensor()])  #单目标检测时候datase图像的归一化是全部一步一步写得  

class yolov3_detect:
    def __init__(self):
        self.pt_path = r"best_loss_60000.pt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = My_Yolov3()
        if os.path.exists(self.pt_path):
            self.net.load_state_dict(torch.load(self.pt_path,map_location=self.device))
            self.net.to(self.device)
            self.net.eval()
        else:
             print("权重路径错误")
    def process(self,img_path):
        # img_data= Image.open(img_path).convert("RGB")
        # img_tensor = tr_tranform(img_data) #CHW-归一化的操作
        
        
        img_pil = Image.open(img_path).convert("RGB")
        img_pil_resize = img_pil.resize((416,416))
        img_tensor = transforms.ToTensor()(img_pil_resize)
        img_tensor = torch.unsqueeze(img_tensor,dim= 0)
        img_tensor = img_tensor.to(self.device)
        return img_pil,img_tensor

    def postprocess(self,preds,threshold=0.70):
        bboxes = []
        for pred in preds:
            # pred：torch.Size([1, 27, 13, 13])
            #torch.Size([1, 13, 13, 3*9])
            # torch.Size([1, 13, 13, 3, 9])
            pred_re = torch.permute(pred,dims = (0,2,3,1))
            pred_re = torch.reshape(pred_re,shape=(pred_re.size(0),pred_re.size(1),pred_re.size(2),3 ,-1))
            feature_size = pred_re.size(1)
            
            recefild = cfg.IMG_SIZE/feature_size
            #获取mask-每个维度索引
            pred_mask=torch.sigmoid(pred_re[:,:,:,:,0])>threshold #torch.Size([1, 13, 13, 3]) nhwc
            #获取每个维度的索引
            #(tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0'),
            #tensor([6, 6, 6, 7, 7, 7, 7, 7, 7], device='cuda:0'),
            # tensor([5, 5, 5, 5, 5, 5, 6, 6, 6], device='cuda:0'), 
            # tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], device='cuda:0'))
            
            pred_idx = torch.where(pred_mask)
            print(pred_idx)
            
            vectors = pred_re[pred_idx]# n,9
            print(vectors)
            conf_ = torch.sigmoid(vectors[:,0]) 
            #cx,cy
            cx = (torch.sigmoid(vectors[:,1])+pred_idx[2])*recefild
            cy  =(torch.sigmoid(vectors[:,2])+pred_idx[1])*recefild
            #w,h
            anchors = torch.tensor(cfg.ANCHORS_GROUPS[feature_size]).to(self.device)
            anchors = anchors[pred_idx[3]]
            w = torch.exp(vectors[:,3])*anchors[:,0]
            h = torch.exp(vectors[:,4])*anchors[:,1]
            #类别
            cls_ = torch.argmax(vectors[:,5:],dim=1)
           

            xmin = cx-w/2
            xmax = cx+w/2
            ymin =cy-h/2
            ymax = cy+h/2
            bbox = torch.stack((xmin,ymin,xmax,ymax,conf_,cls_),dim=1)
            bboxes.append(bbox)
        print(bboxes)
        results_bbox = torch.cat(bboxes,dim = 0)
        return results_bbox
    def draw_img(self,img_pil,bboxs,save_path):
        img_np = np.array(img_pil)
        img_np = cv2.cvtColor(img_np,cv2.COLOR_RGB2BGR)
        for i,bbox in enumerate(bboxs):
            bbox = bbox.detach().cpu().numpy()
            conf = bbox[4]
            xmin,ymin,xmax,ymax = bbox[0:4].astype(np.int64)
            cls_ = bbox[5]
            cv2.rectangle(img_np,(xmin,ymin),(xmax,ymax),(0,0,255),1)
            cv2.putText(img_np,f"conf:{conf:.2f} cls:{cls_}",(xmin,ymin-10),1,0.8,(0,255,0))
            
        os.makedirs("save_img",exist_ok=True)
        cv2.imwrite(save_path,img_np)
    
    def ious(self,bbox,bboxes):
        #交集/并集(xmin,ymin,xmax,ymax,conf_,cls_)
        #bbox:[ 61.2275,  86.5156, 294.4512, 341.5253,   0.9999,   2.0000]
        #bboxes[[ 53.4025,  89.3134, 300.1817, 339.6956,   0.9999,   2.0000],
        # [ 51.9054,  94.2706, 302.8405, 332.7814,   0.9999,   2.0000],
        # [206.0314,  71.2162, 387.5559, 335.1966,   0.9998,   1.0000],
        # [212.5878,  65.2656, 380.6120, 341.0285,   0.9998,   1.0000],
        # [204.9856,  61.3719, 379.8642, 348.9776,   0.9998,   1.0000],
        bbox_area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        bboxes_area= (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])

        l_x = torch.maximum(bbox[0],bboxes[:,0])
        l_y = torch.maximum(bbox[1],bboxes[:,1])
        r_x = torch.minimum(bbox[2],bboxes[:,2])
        r_y = torch.minimum(bbox[3],bboxes[:,3])
        
        inter_w = torch.maximum(r_x-l_x,torch.tensor([0]).to(self.device))
        inter_h = torch.maximum(r_y-l_y,torch.tensor([0]).to(self.device))

        inter_area =  inter_w*inter_h
        union = inter_area/(bbox_area+bboxes_area-inter_area)
        return union
        

    def nms(self,bboxes,thershold=0.35):
        #去除所有iou满足阈值的框   n ,v(xmin,ymin,xmax,ymax,conf_,cls_)
        max_detection= [] 
        #排序
        idx = torch.argsort(bboxes[:,4],descending=True)
        detec_bboxes  = bboxes[idx]
        while detec_bboxes.size(0)>0:
            target_bbox = detec_bboxes[0]
            max_detection.append(target_bbox)
            detec_bboxes = detec_bboxes[1:]
            ious = self.ious(target_bbox,detec_bboxes)
            idx_iou = ious<thershold
            detec_bboxes = detec_bboxes[idx_iou]
        return max_detection 


    def detect(self,img_path):
        #img_0->原图 img_1->输入NCHW NC416,416
        img_0,img_1 = self.process(img_path)  #处理成

        out_13,out26,out52 = self.net(img_1)  
        #后处理：拿到我现在网络检测到的所有的框
        pred_bboxes = self.postprocess((out_13,out26,out52)) 
        #pred_bboxes：nv:n ,v(xmin,ymin,xmax,ymax,conf_,cls_)
        filuter_bbox = self.nms(pred_bboxes)
        save_name = os.path.basename(img_path)
        os.makedirs("save_img",exist_ok=True)
        self.draw_img(img_0,filuter_bbox,f"save_img/{save_name}")
        
        
       

if __name__=="__main__":
    img_dir = r"D:\PycharmProjects\20260717\little_data\images"
    detector  = yolov3_detect()
    for img_name in os.listdir(img_dir):
      
        img_path = os.path.join(img_dir,img_name)
        detector.detect(img_path)