import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def myconvolve(img, kernel, cval=0):
    """
    This function performs convolution of an (single-channel) image and kernel.
    The original image is padded with constant value (cval)


    :img:       Single channel image with which the kernel will be convolved    (np array NxN)
    :kernel:    Kernel to convolve with the image                               (np array 3x3)
    :cval:      Constant value that will be padded around the original single channel
                image to avoid OutOfBounds exception, 0 by default              (int)
    
    :returns:   Convolved image with stripped padding (same size as input image)    
    """
    ret = np.zeros(img.shape, dtype=img.dtype)          
    img = np.pad(img, (1,1), constant_values=(cval))    #padding the outside of image with cval
    rowids = [-1,-1,-1,0,0,0,1,1,1]                     #indices for going through kernel left to right,
    colids = [-1,0,1,-1,0,1,-1,0,1]                     #top to bottom
    for r in range(1,img.shape[0]-1):                   #row by row, except topmost and bottommost
        for c in range(1,img.shape[1]-1):               #column by column, except leftmost and rightmost
            for i,j in zip(rowids, colids):             #go through the whole kernel    
                ret[r-1, c-1] += img[r-i, c-j]*kernel[1+i, 1+j]     #convolution operation as descriped in wiki
    return ret

def applyFilter(filename, kernel, title=""):
    """
    Wrapper for 'myconvolve' function which loads the image by filename and applies said filter (kernel) to it
    This function applies the filter to each channel separately
    If the image has 4 channels, the last one is ignored, as it most likely is alpha channel

    
    :filename:      Filename of the image we wish to apply a filter to  (string)
    :kernel:        3x3 filter which will be applied to the image       (np.array, shape 3,3)
    :title:         The title of the figure                             (string)

    :returns:       The image with applied filter
    """
    assert kernel.shape == (3,3)    #only 3x3 kernel supported
    img = mpimg.imread(filename)    #read image from file    
    img = img.copy()                #original image is read only
    f , ax = plt.subplots(1, 3)     #There will be 3 images in the final figure to see side by side
    f.suptitle(title)               
    numchannels = 1 if len(img.shape) == 2 else img.shape[2]
    if(numchannels == 4):
        #if there are 4 channels, ignore the last one, as it most likely is alphachannel
        img = img[:, :, 0:3]
        numchannels = 3
    if(np.min(img) >= 0.0 and np.max(img) <= 1.0):
        #convert float-type image to integer type image
        img = img*255.0
    img = img.astype(np.int16)
    channels = np.zeros(img.shape, dtype=np.int16)      #output array for my convolution
    scchannels = np.zeros(img.shape, dtype=np.int16)    #output array for scipy convolution    
    for i in range(numchannels):        #Apply filter to all channels separately
        if(numchannels == 1):
            newimg = myconvolve(img, kernel)                            #my convolution
            newimg2 = ndimage.convolve(img, kernel, mode='constant')    #scipy for comparrison
            channels = newimg       #as we have only 1 channel, it is the whole final image
            scchannels = newimg2    
        else:
            newimg = myconvolve(img[:, :, i], kernel)
            channels[:, :, i] = newimg      #add channel with applied filter to output image
            newimg2 = ndimage.convolve(img[:, :, i], kernel, mode='constant')
            scchannels[:, :, i] = newimg2
    newimg = channels
    newimg2 = scchannels
    #setup figure and plots
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[1].imshow(newimg)
    ax[1].set_title("My convolution")
    ax[2].imshow(newimg2)
    ax[2].set_title("Scipy convolution")
    plt.show()
    return newimg


if __name__ == "__main__":
    filename = "img.png"
    identity = np.array([0,0,0,0,1,0,0,0,0]).reshape(3,3)
    sharpen = np.array([0,-1,0,-1,5,-1,0,-1,0]).reshape(3,3)
    edge1 = np.array([1,0,-1,0,0,0,-1,0,1]).reshape(3,3)
    edge2 = np.array([1,0,-1,0,4,0,-1,0,1]).reshape(3,3)
    edge3 = np.array([-1,-1,-1,-1,8,-1,-1,-1,-1]).reshape(3,3)
    #applyFilter(filename, identity, "Identity")
    #applyFilter(filename, sharpen, "Sharpen")
    applyFilter(filename, edge1, "Edge Detection 1")
    #applyFilter(filename, edge2, "Edge Detection 2")
    #applyFilter(filename, edge3, "Edge Detection 3")

