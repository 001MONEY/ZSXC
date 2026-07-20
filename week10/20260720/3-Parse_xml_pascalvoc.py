import xml.etree.ElementTree as ET
import cv2
import os

# 解析xml

def open_xml(path):

    tree = ET.parse(path)
    root = tree.getroot()

    # 获取图片的尺寸
    size = root.find('size')
    w = size.find('width').text
    h = size.findtext('height')
    # print(f"w = {w}, h = {h}")
    f = open(f"{root_dir}/Parse_label.txt", 'a', encoding='utf-8')  # 打开txt文件
    f.writelines("images/%s.jpg " % j)  # 将"images/%s.jpg" % j写入txt文件
    for ele in root.iter("object"):
        cls_name = ele.find('name').text
        # if cls_name == '人':  # 判断名字和分类是否相同
        #     cls_num = 0  # 给每类编号
        # elif cls_name == '猫':
        #     cls_num = 1
        # elif cls_name == '狗':
        #     cls_num = 2
        # else:
        #     cls_num = 3
        if cls_name == 'hongjinyu':  # 判断名字和分类是否相同
            cls_num = 0  # 给每类编号
        elif cls_name == 'heijinyu':
            cls_num = 1
        elif cls_name == 'baijinyu':
            cls_num = 2
       
        bndbox = ele.find('bndbox')
        x1 = int(bndbox.findtext('xmin'))
        y1 = int(bndbox.findtext('ymin'))
        x2 = int(bndbox.findtext('xmax'))
        y2 = int(bndbox.findtext('ymax'))

        box_w = x2 - x1  # 计算框的宽高
        box_h = y2 - y1
        cx = int(x1 + box_w / 2)  # 计算框的中心点坐标并转成int类型
        cy = int(y1 + box_h / 2)
        print(cls_name, (cx, cy, box_w, box_h))
        f.writelines("{} {} {} {} {} \t".format(cls_num, cx, cy, box_w, box_h))  # 按行将起写入txt文件
    f.writelines("\n")  # 换行
    f.flush()  #
    # flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入。
    # 一般情况下，文件关闭后会自动刷新缓冲区，但有时你需要在关闭前刷新它，这时就可以使用 flush() 方法。
    f.close()


if __name__ == '__main__':
    # open_xml()
    root_dir = r"D:\PycharmProjects\20260720\xiaojinyu"
    dirpath = os.path.join(root_dir,'frame_out_416_voc')  # 文件地址
    file_name_list = os.listdir(dirpath)  # 获取文件地址下的所有文件名称
    for file_name in file_name_list:  # 循环列表中的所有文件名
        j = file_name.split('.')[0]
        xml_path = os.path.join(dirpath, file_name)  # 拼接文件地址
        open_xml(xml_path)