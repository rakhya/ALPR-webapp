# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:06:10 2018

@author: prava
"""
import os
cur_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cur_dir,"test_images")
i = 1
for file_name in os.listdir(data_dir):
	os.rename(os.path.join(data_dir,file_name), os.path.join(data_dir,"image{}.jpg".format(i)))
	i +=1
