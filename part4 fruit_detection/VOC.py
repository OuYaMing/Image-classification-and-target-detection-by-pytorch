"""
File:VOC.py
"""

import sys
import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil
import VOCOperationLibrary as vol

class VOC(object):
    def __init__(self, dataset_anno, dataset_img=None, num_class=None, datasetdir=None):
        if os.path.exists(dataset_anno) == False:
            raise  FileNotFoundError
        self.dataset = datasetdir
        self.dataset_anno = dataset_anno
        self.dataset_img = dataset_img
        self.num_class = num_class
        self.dirname = os.path.dirname(self.dataset_anno)
        self.listanno = self._listanno()

    def _listanno(self, annodir=None):
        """return the list of all above of annotation file"""
        if annodir == None:
            annodir = self.dataset_anno
        return os.listdir(annodir)

    def _lowextension(self, imgdir=None):
        return

    def _listimg(self, imgdir=None):
        """return the list of all above of image file"""
        if self.dataset_img == None:
            if imgdir == None:
                print("you should give a image path of dataset in creating VOC class!")
                raise FileNotFoundError
            else:
                return os.listdir(imgdir)
        else:
            return os.listdir(self.dataset_img)

    def _ParseAnnos(self, annodir=None):
        """
        return the information of all above of annotation in this dataset_anno,
        format: a list of dictionary, include file name, annotation, size
        ([{'file', 'info', 'size'}])
        annotation is a list, [cls, xmin, ymin, xmax, ymax]
        size if a tuple, (weight, height)
        """
        annos = []
        if annodir == None:
            annodir = self.dataset_anno
            annolist = self.listanno
        else:
            annolist = self._listanno(annodir)
        for annofile in annolist:
            if annofile[-4:] != ".xml":
                continue
            annotation = vol._parseannotation(os.path.join(annodir, annofile))
            annos.append({'file': annofile, 'info': annotation[0], 'size': annotation[1]})
        return annos

    def _DelAnnotations(self, delclass, annodir=None):
        """
        Delete specific cls
        Precondition:delclass-a list of what's annotaion name you want to delete
        """
        if delclass == None:
            return
        if annodir== None:
            annodir = self.dataset_anno
        annolist = self._listanno(annodir) 
        for annofile in annolist:
            vol._deletesinglefile(os.path.join(annodir, annofile), delclass)

    def _ChangeAnnotation(self, oldcls, newcls, annodir=None):
        """
        Change class name.
        Precondition:
                    oldcls:old class name,string
                    newcls:new class name,string
                    annodir:annotation file direction, if it is None,use self.dataset_dir(init value)
        """
        if annodir == None:
            annodir = self.dataset_anno
        annolist = self._listanno(annodir)
        for annofile in annolist:
            vol._changeone(os.path.join(annodir,annofile), oldcls, newcls)

    def _Crop(self, imgdir, cropdir, annos=None):
        """
        To crop all the box region of object in dataset
        """
        if annos == None:
            annos = self._ParseAnnos()
        total = len(annos)
       
        for num, annotation in enumerate(annos):
            annofile = annotation['file']
            print(annofile)
            print(imgdir+annofile[:-4]+'.jpg')
            if os.path.exists(imgdir+annofile[:-4]+'.jpg') == False:
                raise FileNotFoundError
            pil_im = Image.open(imgdir+annofile[:-4]+'.jpg') 
            for i, obj in enumerate(annotation['info']):
                obj_class = obj[0]
                obj_box = tuple(obj[1:5])
                if os.path.exists(cropdir+obj_class) == False:
                    os.mkdir(cropdir+obj_class)
                region = pil_im.crop(obj_box)
                pil_region = Image.fromarray(np.uint8(region))
                pil_region.save(os.path.join(cropdir+obj_class, 
                                annofile[:-4]+'_'+str(i)+'.jpg'))
            process = int(num*100 / total)
            s1 = "\r%d%%[%s%s]"%(process,"*"*process," "*(100-process))
            s2 = "\r%d%%[%s]"%(100,"*"*100)
            sys.stdout.write(s1)
            sys.stdout.flush()
        sys.stdout.write(s2)
        sys.stdout.flush()
        print('')
        print("crop is completed!")
    
    def _Countobject(self, annofile=None):
        """
        Count the label numbers of every class, and print it
        Precondition: annofile-the direction of xml file
        """
        if annofile == None:
            annofile = self.dataset_anno
        annoparse = self._ParseAnnos(annofile)
        count = {}
        for anno in annoparse:
            for obj in anno['info']:
                if obj[0] in count:
                    count[obj[0]] +=1
                else:
                    count[obj[0]] = 1
        for c in count.items():
            print("{}: {}".format(c[0], c[1]))
        return count

    def _DisplayDirectObjec(self):
        """
        To display what's box you want to display.
        """
        imglist = self._listimg()
        print("input what object you want display, space between numbers")
        parseannos = self._ParseAnnos()
        for i, annos in enumerate(parseannos):
            print("file name: {0}".format(annos['file'][:-4]))
            if annos['info'] == []:
                print("This image don't have annotation, so programme step it and go on!")
                continue
            for j, objs in enumerate(annos['info']):
                print('''({}): cls={}, \
                    box=[{:0>4d}, {:0>4d}, {:0>4d}, {:0>4d}]'''.format(
                    j, objs[0], objs[1], objs[2], objs[3], objs[4]
                ))
            inputstr = input()
            numbers = [int(x) for x in inputstr.split(' ')]
            self._displayone(annos['info'], annos['file'], numbers)
        
    def _displayone(self, objs, annofile, nums):
        """
        display the annotation's box of one image
        Precondition: objs-the box information
                      annofile-annotation file name
                      nums-the object number of annotation which you want display
        """
        im = Image.open(self.dataset_img + annofile[:-4] + '.jpg')
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        for i, obj in enumerate(objs):
            if i in nums:
                bbox = obj[1:]
                ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
                        )
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s}'.format(obj[0]),
                        bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=14, color='white')

        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.show()


    def _Mergeannotation(self, newdataset, olddataset=None):
        """
        Merge two dataset anntation information.
        Precondition:
                    newdataset:one dataset annotation path
                    olddataset:one dataset annotation path 
                               and save in this path
        """
        if olddataset == None:
            olddataset = self.dataset_anno
        annolist1 = os.listdir(olddataset)
        annolist2 = os.listdir(newdataset)
        for anno in annolist2:
            if anno in annolist1:
                print(anno)
                vol._mergeone(olddataset+anno, newdataset+anno)
            else:
                shutil.copy(newdataset+anno, olddataset+anno)

    def _Resize(self, newsize, annodir=None, imgdir=None):
        """
        Resize the dataset, include resize all the image into newsize,
        and correct the annotation information.
        Precondition:
                    newsize:the newsize of image
                    annodir:annotation direction
                    imgdir:image direction
        """
        if annodir == None:
            annodir = self.dataset_anno
        if imgdir == None:
            imgdir = self.dataset_img
            if imgdir == None:
                print('Resize operation need a image direction!')
                return
        annolist = self._listanno(annodir)
        imglist = self._listimg(imgdir)
        annos = self._ParseAnnos(annodir)
        total = len(annolist)
        for num, f in enumerate(annolist):
            anno_path = os.path.join(annodir, f)
            img_path = os.path.join(imgdir, f)[:-4] + '.jpg'
            img = Image.open(img_path)
            img = img.resize(newsize)
            img.save(img_path, 'jpeg', quality=95)
            img.close()
            vol._changeone(anno_path, None, None, newsize)
            process = int(num*100 / total)
            s1 = "\r%d%%[%s%s]"%(process,"*"*process," "*(100-process))
            s2 = "\r%d%%[%s]"%(100,"*"*100)
            sys.stdout.write(s1)
            sys.stdout.flush()
        sys.stdout.write(s2)
        sys.stdout.flush()
        print('')
        print('Resize is complete!')

    def _Splitdataset(self, traintxt, savedir, annodir=None, imgdir=None):
        """
        Split the dataset into train set and test set, according the train.txt.
        Precondition:
                    traintxt:train.txt which include the train set file name
                    savedir:save direction
                    annodir:dataset annotation direction
                    imgdir:dataset image direction
        Result:
            make four direction, trainAnnotations(storage train set's xml file)
                                 trainJPEGImages(storage train set's image file)
                                 testAnnotations(storage test set's xml file)
                                 testJPEGImages(storage test set's image file)
        """
        if annodir == None:
            annodir = self.dataset_anno
        if imgdir == None:
            if self.dataset_img == None:
                print("Please give the path of image!")
            else:
                imgdir = self.dataset_img
        annolist = self._listanno(annodir)
        f = open(traintxt, 'r')
        trainlist = f.readlines()
        f.close()
        train_xml_path = os.path.join(savedir, 'trainAnnotations')
        trian_img_path = os.path.join(savedir, 'trainJPEGImages')
        test_xml_path = os.path.join(savedir, 'testAnnotations')
        test_img_path = os.path.join(savedir, 'testJPEGImages')
        if os.path.exists(train_xml_path) == False:
            os.mkdir(train_xml_path)
        if os.path.exists(trian_img_path) == False:
            os.mkdir(trian_img_path)
        if os.path.exists(test_xml_path) == False:
            os.mkdir(train_xml_path)
        if os.path.exists(test_img_path) == False:
            os.mkdir(test_img_path)
        for i in range(len(trainlist)):
            trainlist[i] = trainlist[i].replace('\n', '')
            annolist.remove(trainlist[i])
        testlist = annolist
        
        self._Copy(trainlist, annodir, imgdir, train_xml_path, trian_img_path)
        self._Copy(trainlist, annodir, imgdir, test_xml_path, test_img_path)    

    def _Copy(self, xml_list, from_xml_path, from_img_path, save_xml_dir, save_img_dir):
        """
        Copy the xml file and image file from dataset to save_xml_dir and save_img_dir.
        Precondition:
                    xml_list:a list of xml file name 
                    from_xml_path:original xml direction
                    from_img_path:original image direction
                    save_xml_dir:to save xml direction
                    save_img_dir:to save image direction
        """
        for i, xml in enumerate(xml_list):
            shutil.copyfile(os.path.join(from_xml_path, xml), 
                        os.path.join(save_xml_dir, xml))
            shutil.copyfile(os.path.join(from_img_path, xml)[:-4] + '.jpg', 
                        os.path.join(save_img_dir, xml)[:-4] + '.jpg')

    def _Find(self, cls, annodir=None):
        """
        Find files of the direction class object.
        Return a list of files name.
        Precondition:
        cls: a list of class, example:['dog', 'cat']
        annodir: the xml files direction
        """
        if annodir == None:
            annodir = self.dataset_anno
        
        annolist = self._listanno(annodir)
        xml_files = []
        for anno in annolist:
            xml = vol._find_one(os.path.join(annodir, anno), cls)
            if xml != None:
                xml_files.append(xml)
        return xml_files

    def _FindandCopy(self, cls, save_xml_path, save_img_path, annodir=None, imgdir=None):
        """
        Find files of the direction class object and copy them.
        Precondition:
                    xml_list:a list of xml file name 
                    annodir:the dataset xml files direction
                    imgdir:the dataset image files direction
                    save_xml_dir:to save xml direction
                    save_img_dir:to save image direction
        """
        if imgdir == None:
            imgdir = self.dataset_img
            if imgdir == None:
                print('Copy operation need a image direction!')
                return
        if annodir == None:
            annodir = self.dataset_anno
        xml_files = self._Find(cls, annodir)
        print(xml_files)
        self._Copy(xml_files, annodir, imgdir, save_xml_path, save_img_path)

v = VOC('F:/My_py/pytorch/a-PyTorch-Tutorial-to-Object-Detection-master/underwater_dataset/train_data/Annotations/', 
       'F:/My_py/pytorch/a-PyTorch-Tutorial-to-Object-Detection-master/underwater_dataset/train_data/JPEGImages')
#print('11',v._ParseAnnos())
#v._Crop('F:/My_py/pytorch/a-PyTorch-Tutorial-to-Object-Detection-master/underwater_dataset_100/train_data/JPEGImages/', 'F:/My_py/pytorch/a-PyTorch-Tutorial-to-Object-Detection-master/underwater_dataset_100/train_data/crop/')
#v._DelAnnotations(['123', '234'])
#v._DisplayDirectObjec()
size = (300, 167)
v._Resize(size)
#v._Mergeannotation('C:/Users/91279/Desktop/xml/', 'F:/xml/')
#v._DelAnnotations(['123'])
#cls = ['shockproof hammer deformation', 'shockproof hammer intersection', 'grading ring damage', 'shielded ring corrosion']
#v._FindandCopy(cls, 'F:/数据集/20190122输电线路主要缺陷优化数据集/aaaa/', 'F:/数据集/20190122输电线路主要缺陷优化数据集/bbbb/')