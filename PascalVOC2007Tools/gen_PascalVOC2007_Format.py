import os
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm
import cv2
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import random


class PascalVOC2007():
    def __init__(self,myImageFolder,TargetImageFolder):
        self.ImageFolder = myImageFolder
        self.TargetImageFolder = TargetImageFolder
        self.ImageList = os.listdir(self.ImageFolder)
        self.Old2NewName_dict = {}
        self.Image2XML_dict = {}

    def rename(self):
        for idx,ImageName in enumerate(tqdm(self.ImageList)):
            OldPath = os.path.join(self.ImageFolder,ImageName)
            NewPath = os.path.join(self.TargetImageFolder,str(idx+1).zfill(6)+'.jpg')
            shutil.copyfile(OldPath,NewPath)
            self.Old2NewName_dict[ImageName] = str(idx+1).zfill(6)+'.jpg'

    def gen_XML(self,train_csv_path):
        train_df = pd.read_csv(train_csv_path)
        ImageList = list(set(train_df['ID'].values.tolist()))
        for Image in tqdm(ImageList):
            ImageInfo_df = train_df[train_df['ID']==Image]
            ImageDetection_list = ImageInfo_df[' Detection'].values.tolist()
            Bbox_dict = {}
            for idx,ImageDetection in enumerate(ImageDetection_list):
                xmin,ymin,xmax,ymax = ImageDetection.split(' ')
                xmin,ymin,xmax,ymax = int(xmin),int(ymin),int(xmax),int(ymax)
                Bbox_dict[idx] = {'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax}
            NewImage = self.Old2NewName_dict[Image]
            self.Image2XML_dict[NewImage] = Bbox_dict
            img = cv2.imread(os.path.join(self.TargetImageFolder,NewImage))
            w,h = img.shape[1],img.shape[0]
            iter = NewImage.split('.')[0]
            self.write_XML(NewImage,Bbox_dict,w,h,iter)


    def write_XML(self,ImageName,Bbox_dict,w,h,iter):
         root=Element("annotation")
         folder=SubElement(root,"folder")#1
         folder.text=self.TargetImageFolder

         filename=SubElement(root,"filename")#1
         filename.text=ImageName

         path=SubElement(root,"path")#1
         path.text = os.path.join(self.TargetImageFolder,ImageName)

         source=SubElement(root,"source")#1
         database=SubElement(source,"database")#2
         database.text="MyDataset"

         size=SubElement(root,"size")#1
         width=SubElement(size,"width")#2
         height=SubElement(size,"height")#2
         depth=SubElement(size,"depth")#2
         width.text=str(w)
         height.text=str(h)
         depth.text='3'

         segmented=SubElement(root,"segmented")#1
         segmented.text='0'

         for k in Bbox_dict.keys():
             object=SubElement(root,"object")#1
             name=SubElement(object,"name")#2
             name.text="gangjin"
             pose=SubElement(object,"pose")#2
             pose.text="Unspecified"
             truncated=SubElement(object,"truncated")#2
             truncated.text='0'
             difficult=SubElement(object,"difficult")#2
             difficult.text='0'
             bndbox=SubElement(object,"bndbox")#2
             xmin=SubElement(bndbox,"xmin")#3
             ymin=SubElement(bndbox,"ymin")#3
             xmax=SubElement(bndbox,"xmax")#3
             ymax=SubElement(bndbox,"ymax")#3
             xmin.text=str(Bbox_dict[k]['xmin'])
             ymin.text=str(Bbox_dict[k]['ymin'])
             xmax.text=str(Bbox_dict[k]['xmax'])
             ymax.text=str(Bbox_dict[k]['ymax'])

         #tree = ET.ElementTree(root)
         xml = tostring(root, pretty_print=True)  #格式化显示，该换行的换行
         dom = parseString(xml)
         xml_name = '/home/lab404/Documents/PyData/Pascal VOC2007 For GangJin/Annotations/'+iter+'.xml'
         with open(xml_name , 'w') as f:
            dom.writexml(f, addindent='\t')


    def gen_maintxt(self,XMLfolder,MainTXTfolder,AllTrainFlag):
        if AllTrainFlag:         # is the image in JPEGImages Folder only for train?
            trainval_percent = 1
            train_percent = 0.9
        else:
            trainval_percent = 0.5
            train_percent = 0.9

        total_xml = os.listdir(XMLfolder)
        num=len(total_xml)
        list=range(num)
        tv=int(num*trainval_percent)
        tr=int(tv*train_percent)
        random.seed(2019)
        trainval= random.sample(list,tv)
        train=random.sample(trainval,tr)
        ftrainval = open(os.path.join(MainTXTfolder,'Main/trainval.txt'), 'w+')
        ftest = open(os.path.join(MainTXTfolder,'Main/test.txt'), 'w+')
        ftrain = open(os.path.join(MainTXTfolder,'Main/train.txt'),'w+')
        fval = open(os.path.join(MainTXTfolder,'Main/val.txt'), 'w+')

        for i  in list:
            name=total_xml[i][:-4]+'\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)
        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest .close()


#Main
myImageFolder = '/home/lab404/Documents/PyData/GangJin/train_dataset/'
TargetImageFolder = '/home/lab404/Documents/PyData/Pascal VOC2007 For GangJin/JPEGImages/'
train_csv_path = '/home/lab404/Documents/PyData/GangJin/train_labels.csv'
XMLfolder = '/home/lab404/Documents/PyData/Pascal VOC2007 For GangJin/Annotations'
MainTXTfolder = '/home/lab404/Documents/PyData/Pascal VOC2007 For GangJin/IMageSets/'
AllTrainFlag = 1

PascalVOC2007 = PascalVOC2007(myImageFolder,TargetImageFolder)
PascalVOC2007.rename()
PascalVOC2007.gen_XML(train_csv_path)
PascalVOC2007.gen_maintxt(XMLfolder,MainTXTfolder,AllTrainFlag)
