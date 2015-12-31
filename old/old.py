__author__ = 'vozka'
import numpy as np
import cv2
import SimpleITK as sitk
from scipy import ndimage
from scipy import signal
from matplotlib import pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import timeit #db
import sys

'''
def main(argv):
    if len(argv) != 5:
        print "expecting 5 parameters: image name, number of dimensions (2/3), sigma of foreground gaussian, " \
              "sigma of background gaussian and number of steps"



if __name__ == "__main__":
   main(sys.argv[1:])
   '''

'''
def multiscale2DBG(image, sigmaf, sigmab, steps):
    '''Implements multiscale filtering: for each step the image is blurred, downsampled,
    has its lineness function computed and then upsampled and blurred again. The result is compared to the
    final output and larger pixel intensity values are used. Probably incorrect'''

    image_out = np.zeros_like(image) #fill output image with zeros
    for i in range(steps):
        kernel = biGaussianKernel2D(sigmaf * (2**i), sigmab * (2**i)) #generate the bigaussian kernel for each step
        img_filtered = cv2.filter2D(image, -1, kernel)
        img_downscaled = cv2.resize(img_filtered, (np.size(image, 1)/(2**i), np.size(image, 0)/(2**i)), interpolation=cv2.INTER_NEAREST) #downscale the image if appropriate
        imgh = hessian2D(img_downscaled) #compute the hessian
        img_e = eigenvalues2D(imgh) #compute the eigenvalues from hessian
        imge_resized = cv2.resize(img_e, (np.size(image, 1), np.size(image, 0)), interpolation=cv2.INTER_NEAREST) #upscale back to normal
        image_out = np.maximum(image_out, cv2.filter2D(imge_resized, -1, kernel)) #blur again, compare to output image and take the higher intensity pixels

    max = np.amax(image_out)
    return ((image_out/max)*255).astype(int) #normalize the image to 0-255 and return

img3d = sitk.ReadImage('zmenseno.mha')
array3d = sitk.GetArrayFromImage(img3d)

start_time = timeit.default_timer()
hessian3d = hessian3D(array3d)
print "hessian", timeit.default_timer() - start_time
start_time = timeit.default_timer()
eigenval = eigenvalues3D(hessian3d)
print "eigenvalues", timeit.default_timer() - start_time

start_time = timeit.default_timer()
result = lineness(eigenval)
print "lineness", timeit.default_timer() - start_time

max = np.amax(result)
sitk_img = sitk.GetImageFromArray(((result/max)*255).astype(int))
sitk.WriteImage(sitk_img, os.path.join("./", 'linenesss.mha'))
'''

#img3dfft_array = signal.fftconvolve(array3d, kernel3d, mode="same")


#img_T1_255 = sitk.Cast(sitk.RescaleIntensity(img_T1), sitk.sitkUInt8)
#img3df = sitk.Convolution(img3d, kernel3d)


img = cv2.imread('nazghul-small.jpg', 0)
dst = multiscale2DBG(img, 2, 1, 2)


#imgf = cv2.filter2D(img, -1, kernel)

#start_time = timeit.default_timer()
#imgh = hessian2D(img)
#print "hessian", timeit.default_timer() - start_time
#start_time = timeit.default_timer()
#imge = eigenvalues2D(imgh)
#print "eigenvalues", timeit.default_timer() - start_time
#print np.amax(imge), np.amin(imge)
#max = np.amax(imge)
#imge = ((imge/max)*255).astype(int)

#cv2.imshow('image', np.uint8(imge))
cv2.imwrite("nazghul-small_bg2.jpg", dst)
#sitk_img = sitk.GetImageFromArray(eigenval)
#sitk.WriteImage(sitk_img, os.path.join("./", 'eigenval.mha'))
#kernel_sitk = sitk.GetImageFromArray(kernel3d)
#sitk.WriteImage(kernel_sitk, os.path.join("./", 'hessian.mha'))

#plt.imshow(dst)
#plt.imshow(kernel, interpolation='nearest')
#plt.show()


#cv2.waitKey(0)
#cv2.destroyAllWindows()