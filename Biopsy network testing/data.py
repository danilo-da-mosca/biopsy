import os
import numpy as np
from skimage.io import imsave, imread, imshow
data_path=""
train_data_path = os.path.join(data_path, 'train')
img = imread(os.path.join(train_data_path, "1.tif"), as_gray=True)
img_mask = imread(os.path.join(train_data_path, "1_mask.tif"), as_gray=True)
imshow(img_mask)
height, width = img.shape
print(np.unique(img_mask)/255)
thresh_indices=[0,0.5,1]
mask = np.zeros_like(img_mask)
#for indice in range(3):
#print(thresh_indices[indice])
idx0=np.where(img_mask==0)
idx1=np.where((img_mask>0) & (img_mask<255))
idx2=np.where((img_mask==255))
mask[idx1]=0.5*255
mask[idx2]=255
n=3
real_mask = np.zeros([mask.shape[0], mask.shape[1], n])
for class_ in range(n-1):
    real_mask[:,:,class_] = (mask[:,:] >= thresh_indices[class_]) * (mask[:,:] <  thresh_indices[class_+1])
real_mask[:,:,-1] = (mask[:,:] >= thresh_indices[-1])
real_mask = real_mask.astype(np.uint8)
print (img.shape)
print(real_mask.shape)
np.save('imgs_train.npy', img)
np.save('imgs_mask_train.npy', real_mask)
