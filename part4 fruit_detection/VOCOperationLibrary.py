"""
File:VOCOperationLibrary.py
"""

import sys
import os
import xml.etree.ElementTree as ET
import numpy as np
import random
import shutil

def _parseannotation(annofile):
    """
    return a array include class name, box([cls, xmin, ymin, xmax, ymax])
    and a tuple include the size of object((weight, height))
    """
    if os.path.exists(annofile) == False:
        raise FileNotFoundError
    tree = ET.parse(annofile)
    annos = []

    height = 2160
    weight = 3840
    '''
    for annoobject in tree.iter():
        if 'size' in annoobject.tag:
            for element in list(annoobject):
                if 'height' in element.tag:
                    height = int(element.text)
                    print('11111111111111111111111111')
                if 'width' in element.tag:
                    weight = int(element.text)
    '''
    for annoobject in tree.iter():
        if 'object' in annoobject.tag:
            for element in list(annoobject):
                if 'name' in element.tag:
                #    print('2222222222222222')
                    name = element.text
                if 'bndbox' in element.tag:
                    for size in list(element):
                        if 'xmin' in size.tag:
                            xmin = size.text
                        if 'ymin' in size.tag:
                            ymin = size.text
                        if 'xmax' in size.tag:
                            xmax = size.text
                        if 'ymax' in size.tag:
                            ymax = size.text
                    annos.append([name, int(xmin), int(ymin), int(xmax), int(ymax)])
    return annos, (weight, height)

def _deletesinglefile(annofile, delclass):
    if os.path.exists(annofile) == False:
        raise FileNotFoundError
    tree = ET.parse(annofile)
    root = tree.getroot()
    annos = [anno for anno in root.iter()]
    for i, anno in enumerate(annos):
        if 'object' in anno.tag:
            for element in list(anno):
                if 'name' in element.tag:
                    if element.text in delclass:
                        root.remove(annos[i])
                        print(os.path.basename(annofile)+' have something deleted')
                break
    tree = ET.ElementTree(root)
    tree.write(annofile, encoding="utf-8", xml_declaration=True)

def _changeone(annofile, oldcls, newcls, newsize=None):
        if os.path.exists(annofile) == False:
            raise FileNotFoundError
        tree = ET.parse(annofile)
        root = tree.getroot()
        annos = [anno for anno in root.iter()]
        
        for i, anno in enumerate(annos):
            '''
            if newsize != None:
                if 'width' in anno.tag:
                    oldwidth = float(anno.text)
                    anno.text = str(newsize[0])
                    sizechangerate_x = newsize[0] / oldwidth
                if 'height' in anno.tag:
                    oldheight = float(anno.text)
                    anno.text = str(newsize[1])
                    sizechangerate_y = newsize[1] / oldheight
            '''
            sizechangerate_x = 0.078125
            sizechangerate_y = 0.077315
            if 'object' in anno.tag:
                for element in list(anno):
                    if oldcls != newcls:
                        if 'name' in element.tag:
                            if element.text == oldcls:
                                element.text = newcls
                                print(os.path.basename(annofile)+' change the class name')
                        break
                    if newsize != None:
                        if 'bndbox' in element.tag:
                            for coordinate in list(element):
                                if 'xmin' in coordinate.tag:
                                    coordinate.text = str(int(int(coordinate.text) * sizechangerate_x))
                                if 'xmax' in coordinate.tag:
                                    coordinate.text = str(int(int(coordinate.text) * sizechangerate_x))
                                if 'ymin' in coordinate.tag:
                                    coordinate.text = str(int(int(coordinate.text) * sizechangerate_y))
                                if 'ymax' in coordinate.tag:
                                    coordinate.text = str(int(int(coordinate.text) * sizechangerate_y))
                        
        tree = ET.ElementTree(root)
        tree.write(annofile, encoding="utf-8", xml_declaration=True)

def _mergeone(anno1, anno2):
        tree = ET.parse(anno1)
        root = tree.getroot()
        annos, size = _parseannotation(anno2)
        if annos == None:
            return
        for annotation in annos:
            appendobj(root, annotation)
        tree.write(anno1, encoding='utf-8', xml_declaration=True)

def appendobj(root, annotation):
    obj = ET.Element('object')
    name = ET.SubElement(obj, 'name')
    name.text = annotation[0]
    pose = ET.SubElement(obj, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(obj, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(obj, 'difficult')
    difficult.text = '0'
    bndbox = ET.SubElement(obj, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = str(annotation[1])
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = str(annotation[2])
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = str(annotation[3])
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(annotation[4])
    root.append(obj)
    return root

def _find_one(annofile, cls):
        if os.path.exists(annofile) == False:
            raise FileNotFoundError

        tree = ET.parse(annofile)
        root = tree.getroot()
        annos = [anno for anno in root.iter()]
        
        for i, anno in enumerate(annos):
            if 'object' in anno.tag:
                for element in list(anno):
                        if 'name' in element.tag:
                            if element.text in cls:
                                return os.path.basename(annofile)
                        break                
        
        