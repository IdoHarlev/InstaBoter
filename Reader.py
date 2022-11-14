import pandas
import numpy
import torch
from PIL import Image
import glob
import os

rootdir ='/Users/ido/Desktop/CoachAI/InstaBoter/rotemsela1'

def get_jpg_from_main_dir(mainroot):
     for subdir, dirs, files in os.walk(mainroot):
          print(subdir)
          jpgFilenamesList = glob.glob(str(subdir) + '/*.jpg')
          for i in jpgFilenamesList:
               image = Image.open(i)
               #image.show()

'''
subdirs = [x[1] for x in os.walk(rootdir)]
print(subdirs[0][0])
'''

