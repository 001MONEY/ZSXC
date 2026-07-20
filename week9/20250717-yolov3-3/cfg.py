import torch
'自定义的建议框'
ANCHORS_GROUPS = {
    13: [[360, 360], [360, 180], [180, 360]],
    26: [[180, 180], [180, 90], [90, 180]],
    52: [[90, 90], [90, 45], [45, 90]]
}
# ANCHORS_GROUPS = {
#     13: [[198, 113], [156, 224], [311, 248]],
#     26: [[42, 92], [81, 73], [89, 153]],
#     52: [[15, 20], [24, 49], [46, 33]],
# }
device = "cuda:0" if torch.cuda.is_available() else "cpu"
CLASS_NUM = 4
IMG_SIZE  =416
classes = ["bus", "car", "cat","person"]  

model_path  = r"test_01\280.pt"

cof_thresh = 0.5