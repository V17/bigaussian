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


def multiscale2DBG(image, sigmaf, sigmab, step, nsteps):
    """Implements multiscale filtering for 2D images: for each step the image is blurred using accordingly sized
    bigaussian, hessian matrix is computed for each pixel and the largest (absolute) eigenvalue is found. If the
    eigenvalue intensity is larger than the intensity in the output image (initialized with zeros), it is used
    as the output value for that pixel."""

    image = image.astype(np.float)
    image_out = np.zeros_like(image)
    p = sigmab/sigmaf
    for i in range(nsteps):
        stime = timeit.default_timer()

        kernel = biGaussianKernel2D(sigmaf + (i * step), (sigmaf + (i * step))*p) #generate the bigaussian kernel for each step

        print i+1, "- bigaussian kernel generated in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_filtered = ndimage.filters.convolve(image, kernel)

        print i+1, "- image filtered in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_hessian = hessian2D(img_filtered, sigmaf + (i * step))

        print i+1, "- hessian computed in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_e = np.clip(max_eig2D_alt(img_hessian), 0, 255)

        print i+1, "- eigenvalues and lineness computed in", timeit.default_timer() - stime, "s"

        image_out = np.maximum(image_out, img_e) #compare to output image and take the higher intensity pixels

    max = np.amax(image_out)
    image_out *= (255.0 / max)

    return image_out.astype(np.uint8) #normalize the image to 0-255 and return


def multiscale3DBG(image, sigmaf, sigmab, step, nsteps):
    """Implements multiscale filtering for 3D images: for each step the image is blurred using accordingly sized
    bigaussian, hessian matrix is computed for each pixel and the lineness function is computed from its eigenvalues.
    If the lineness intensity is larger than the intensity in the output image (initialized with zeros), it is used
    as the output value for that pixel."""

    image_out = np.zeros_like(image, dtype=np.float32)
    p = sigmab / sigmaf
    for i in range(nsteps):

        stime = timeit.default_timer()

        kernel = biGaussianKernel3D(sigmaf + (i * step), (sigmaf + (i * step))*p)

        print i+1, "- bigaussian kernel generated in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_filtered = ndimage.filters.convolve(image.astype(np.float32), kernel.astype(np.float32))

        print i+1, "- image filtered in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_hessian = hessian3D(img_filtered, sigmaf + (i * step))

        print i+1, "- hessian computed in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_eigenvalues = eigenvalues3D(img_hessian)#.astype(np.float16)

        print i+1, "- eigenvalues computed in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_lineness = lineness3D(img_eigenvalues)#.astype(np.float16)

        print i+1, "- lineness filter response computed in", timeit.default_timer() - stime, "s"

        image_out = np.maximum(image_out, img_lineness)

    max_value = np.amax(image_out)
    #return ((image_out/max_value)*255).astype(np.uint8)
    return image_out#.astype(np.uint16)

def filter3d(imagein, imageout, sigma_foreground=1, sigma_background=0.4, step_size=0.2, number_steps=3):
    """Loads a 3D image, applies the filter, saves the result"""
    img3d = sitk.GetArrayFromImage(sitk.ReadImage(imagein))
    stime = timeit.default_timer()
    dst = multiscale3DBG(img3d, sigma_foreground, sigma_background, step_size, number_steps)
    sitk_img = sitk.GetImageFromArray(dst)
    print "single-thread filter finished in", timeit.default_timer() - stime, "s"
    sitk.WriteImage(sitk_img, os.path.join("./", imageout))


def filter2d(imagein, imageout, sigma_foreground=1, sigma_background=0.4, step_size=0.2, number_steps=3):
    """Loads a 2D image, applies the filter, saves the result"""
    img2d = sitk.ReadImage(imagein)
    array2d = sitk.GetArrayFromImage(img2d)
    if len(array2d.shape) == 3:
        array2d = np.mean(array2d, -1) #converts to grayscale

    stime = timeit.default_timer()
    dst = multiscale2DBG(array2d, sigma_foreground, sigma_background, step_size, number_steps)
    sitk_img2d = sitk.GetImageFromArray(dst)
    print "single-thread filter finished in", timeit.default_timer() - stime, "s"
    sitk.WriteImage(sitk_img2d, os.path.join("./", imageout))


def max_eig2d_pool(hessian):
    pool = multiprocessing.Pool(processes=4)
    eigenvalues = np.array(pool.map(np.linalg.eigvals, hessian))
    sorted_index = np.argsort(np.fabs(eigenvalues), axis=2)
    static_index = np.indices((hessian.shape[0], hessian.shape[1], 2))

    eigenvalues = eigenvalues[static_index[0], static_index[1], sorted_index]
    return (np.transpose(eigenvalues, (2, 0, 1))[1] * (-1)).clip(0)

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