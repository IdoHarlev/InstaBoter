import pandas as pd
import numpy
import torch
from PIL import Image
import glob
import os

rootdir ='/Users/ido/Desktop/CoachAI/InstaBoter/idoharlev'
rootdir1 = '/Users/ido/Desktop/CoachAI/Emitions/test'


def get_jpg_from_main_dir(mainroot):
     subdirs = [x[1] for x in os.walk(mainroot)]
     print(subdirs[0])
     for i,k in enumerate(subdirs[0]):
          root = mainroot + '/' + str(k) + '/*.jpg'
          jpgFilenamesList = glob.glob(root)
          for j in jpgFilenamesList:
               print(j,k)
               image = Image.open(j)
               # image.show()


     '''
     for subdir, dirs, files in os.walk(mainroot):
          print(subdir)

          try:
               jpgFilenamesList = glob.glob(str(subdir) + '/*.jpg')
          except:
               jpgFilenamesList = glob.glob(str(subdir) + '/*.png')
          for i in jpgFilenamesList:
               image = Image.open(i)
               #image.show()
               '''

get_jpg_from_main_dir(rootdir)




