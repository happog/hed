import numpy as np
import scipy.misc
import Image
import os
import cv2
import skimage.exposure as exposure

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/hed/
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# data_root = '../../data/HED-BSDS/'
data_root = '../../data/'
with open(data_root+'test.lst') as f:
    test_lst = f.readlines()
    
test_lst = [data_root+x.strip() for x in test_lst]

im_lst = []
for i in range(0, len(test_lst)):
    im = Image.open(test_lst[i])
    # in_ = np.array(im, dtype=np.uint8)
    in_ = np.array(im, dtype=np.float32)
    if in_.shape[2]>3:
        in_ = in_[:,:,:3] # remove alpha channel
    in_ = in_[:,:,::-1] # rgb to bgr
    # in_=exposure.equalize_adapthist(in_, kernel_size=None, clip_limit=0.2, nbins=64) # enhance
    in_ = in_/np.max(in_)*255.0
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    im_lst.append(in_)

idx = 3
in_ = im_lst[idx]
in_ = in_.transpose((2,0,1)) # HWC to CHW
#remove the following two lines if testing with cpu
# caffe.set_mode_gpu()
# caffe.set_device(0)
# load net
model_root = './'
net = caffe.Net(model_root+'deploy.prototxt', model_root+'hed_pretrained_bsds.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]

merge = (out1+out2+out3+out4+out5+fuse)/6.0
print(np.max(merge))
cv2.imwrite(test_lst[idx][:-4]+"-res.png",merge/np.max(merge)*255)
cv2.imwrite(test_lst[idx][:-4]+"-res-out3.png",out3/np.max(out3)*255)


# process the watershed
from skimage.morphology import watershed, disk
from scipy import ndimage as ndi
bd = int(out3.shape[1]/20)
out3[:,:bd] = 0
out3[:,-bd:-1] = 0
markers = out3 < 0.05 # 0.05
markers = ndi.label(markers)[0]
labels = watershed(out3, markers)
cv2.imwrite(test_lst[idx][:-4]+"-label.png",labels)

# binary processing
uniq = np.unique(labels[:,-1],return_counts=True)
bg = uniq[0][np.argmax(uniq[1])]
binary = 255*(labels!=bg)
cv2.imwrite(test_lst[idx][:-4]+"-bin.png",binary)

# scale_lst = [fuse]
# plot_single_scale(scale_lst, 22)
# scale_lst = [out1, out2, out3, out4, out5]
# plot_single_scale(scale_lst, 10)
