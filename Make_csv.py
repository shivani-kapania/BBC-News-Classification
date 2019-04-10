#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import glob
import os


#l = [pd.read_csv(filename, header=None, sep='\0') for filename in glob.glob("bbc/business/*.txt")]
#df = pd.concat(l, axis=0)

#df.shape



data_folder = "./bbc/"
folders = ["business","entertainment","politics","sport","tech"]



#os.listdir(data_folder)


files = os.listdir('./bbc/business/')


folderslist = [f for f in os.listdir(data_folder) if not f.startswith('.')]


#folderslist


news = []
newstype = []

#traverse each folder in bbc
for folder in folders:
    folder_path = './bbc/'+folder+'/'
    #list all files in a particular news category
    files = os.listdir(folder_path)
    for text_file in files:
        file_path = folder_path + "/" +text_file
        #read contents of a file
        with open(file_path, errors='replace') as f:
            data = f.readlines()
        data = ' '.join(data)
        #append the news article and it's category to two lists
        news.append(data)
        newstype.append(folder)


#len(newstype)


#make a dict of news and its corresponding type
datadict = {'news':news, 'type':newstype}


#write the dictionary to a csv file
df = pd.DataFrame(datadict)
df.to_csv('./bbc.csv')




