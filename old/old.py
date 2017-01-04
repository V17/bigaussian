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


def bg3d(imagein, imageout, sigma_foreground=1, sigma_background=0.4, step_size=0.5, number_steps=3):
    img3d = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join("./input/", imagein)))
    p = sigma_background / sigma_foreground
    for i in range(number_steps):
        kernel = bigaussian_kernel_3d(sigma_foreground + (i * step_size), (sigma_foreground + (i * step_size)) * p)
        img_filtered = ndimage.filters.convolve(img3d, kernel)
        sitk.WriteImage(sitk.GetImageFromArray(img_filtered), os.path.join("./output/", str(i)+imageout))


def poisson(imagein, imageout):
    img3d = sitk.GetArrayFromImage(sitk.ReadImage(imagein)).astype(np.float32)
    print img3d.shape
    maximum = np.amax(img3d)
    minimum = np.amin(img3d)
    print "original:", minimum, maximum
    noise_mask = np.random.poisson((img3d+64.0)/2.347)
    print (img3d+64)/2.347
    print noise_mask
    minimum = np.amin(noise_mask)
    maximum = np.amax(noise_mask)
    print "noise:", minimum, maximum
    output = np.clip((noise_mask*2.0)-64.0, 0, 255).astype(np.uint8)
    minimum = np.amin(output)
    maximum = np.amax(output)
    print "output:", minimum, maximum
    peak = 50
    noisy = np.clip(np.random.poisson(img3d.astype(np.float32)+10) - 10, 0, 255).astype(np.uint8)
    print noisy
    sitk.WriteImage(sitk.GetImageFromArray(output), imageout)


def hausdorff_dist(source, target):
    vol_a = sitk.GetArrayFromImage(sitk.ReadImage(source))
    vol_b = sitk.GetArrayFromImage(sitk.ReadImage(target))
    dist_lst = []
    print len(vol_a)
    for idx in range(len(vol_a)):
        dist_min = 1000.0
        for idx2 in range(len(vol_b)):
            dist = np.linalg.norm(vol_a[idx]-vol_b[idx2])
            if dist_min > dist:
                dist_min = dist
        dist_lst.append(dist_min)
    return np.max(dist_lst)


def hessian2d_multi(imagein, imageout, sigma_foreground=1.0, sigma_background=0.4, step_size=0.5, number_steps=3):
    array2d = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join("./input/", imagein))).astype(np.float)
    if len(array2d.shape) == 3:
        array2d = np.mean(array2d, -1)
    p = sigma_background/sigma_foreground
    #print array2d
    for i in range(number_steps):
        kernel = bigaussian_kernel_2d(sigma_foreground + (i * step_size), sigma_foreground + (i * step_size) * p)
        img_filtered = ndimage.filters.convolve(array2d, kernel)
        #print img_filtered
        img_hessian = hessian2d(img_filtered, sigma_foreground + (i * step_size))
        #print img_hessian
        img_eigen = np.clip(max_eigenvalue_magnitude_2d(img_hessian), 0, 255)
       # print img_eigen
        print np.max(img_eigen), np.min(img_eigen)
        img_eigen = img_eigen.astype(np.uint8)
        print np.max(img_eigen), np.min(img_eigen)
        sitk.WriteImage(sitk.GetImageFromArray(img_eigen.astype(np.uint8)), os.path.join("./output/", "hessian"+str(i)+imageout))

def multiscale3DBG_step(image, kernel, i, sigma, return_dict):
    """Single iteration of 3D bigaussian filter, stores the output in the return_dict list"""

    stime = timeit.default_timer()
    img_resized = np.pad(image, int((kernel.shape[0]/2)), mode='reflect')
    img_filtered = signal.fftconvolve(img_resized, kernel, mode='valid')
    print i+1, "- image filtered with bi-gaussian in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_hessian = hessian3D(img_filtered, sigma)
    print i+1, "- hessian computed in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_eigenvalues = eigenvalues3D(img_hessian).astype(np.float16)
    print i+1, "- eigenvalues computed in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_lineness = lineness_bg_3d(img_eigenvalues).astype(np.float16)
    print i+1, "- lineness filter response computed in", timeit.default_timer() - stime, "s"

    return_dict.append(img_lineness)
    return


def multiscale_frangi_step(image, kernel, i, sigma, return_dict):
    """Single iteration of 3D frangi filter, stores the output in the return_dict list"""

    stime = timeit.default_timer()
    img_resized = np.pad(image, int((kernel.shape[0]/2)), mode='reflect')
    img_filtered = signal.fftconvolve(img_resized, kernel, mode='valid')
    print i+1, "- image filtered with gaussian in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_hessian = hessian3D(img_filtered, sigma)
    print i+1, "- hessian computed in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_eigenvalues = eigenvalues3D(img_hessian).astype(np.float)
    print i+1, "- eigenvalues computed in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_lineness = lineness_frangi_3d(img_eigenvalues).astype(np.float)
    print i+1, "- lineness filter response computed in", timeit.default_timer() - stime, "s"

    return_dict.append(img_lineness)
    return


def multiscale_sato_step(image, kernel, i, sigma, return_dict):
    """Single iteration of 3D frangi filter, stores the output in the return_dict list"""

    stime = timeit.default_timer()
    img_resized = np.pad(image, int((kernel.shape[0]/2)), mode='reflect')
    img_filtered = signal.fftconvolve(img_resized, kernel, mode='valid')
    print i+1, "- image filtered with gaussian in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_hessian = hessian3D(img_filtered, sigma)
    print i+1, "- hessian computed in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_eigenvalues = eigenvalues3D(img_hessian).astype(np.float)
    print i+1, "- eigenvalues computed in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_lineness = lineness_sato_3d(img_eigenvalues).astype(np.float)
    print i+1, "- lineness filter response computed in", timeit.default_timer() - stime, "s"

    return_dict.append(img_lineness)
    return




def sato_filter3d(imagein, imageout, sigma=1.0, step_size=0.5, number_steps=3):
    img3d = sitk.GetArrayFromImage(sitk.ReadImage(imagein)).astype(np.float)
    #max_value = np.amax(img3d)
    #img3d *= (255.0 / max_value)
    stime = timeit.default_timer()
    image_out = np.zeros_like(img3d, dtype=np.float)
    return_list = list()
    for i in range(number_steps):
        multiscale_sato_step(img3d, sigma, i, step_size, return_list)
    for result in return_list:
        image_out = np.maximum(image_out, result)

    #image_out = np.clip(image_out, 0, 255)
    max_value = np.amax(image_out)
    if max_value < 255:
        image_out *= (255.0 / max_value)
    #threshold = rosin_threshold(image_out)
    #mask = image_out > threshold
    #image_out[mask] = 255
    #image_out *= mask
    sitk_img = sitk.GetImageFromArray(image_out.astype(np.uint8))
    print "filter finished in", timeit.default_timer() - stime, "s"
    sitk.WriteImage(sitk_img, os.path.join("./", imageout))


def frangi_filter3d(imagein, imageout, sigma=1.0, step_size=0.5, number_steps=3):
    img3d = sitk.GetArrayFromImage(sitk.ReadImage(imagein)).astype(np.float)
    #max_value = np.amax(img3d)
    #img3d *= (255.0 / max_value)
    stime = timeit.default_timer()
    image_out = np.zeros_like(img3d, dtype=np.float)
    return_list = list()
    for i in range(number_steps):
        multiscale_frangi_step(img3d, sigma, i, step_size, return_list)
    for result in return_list:
        image_out = np.maximum(image_out, result)

    #image_out = np.clip(image_out, 0, 255)
    max_value = np.amax(image_out)
    if max_value < 255:
        image_out *= (255.0 / max_value)
    #threshold = rosin_threshold(image_out)
    #mask = image_out > threshold
    #image_out[mask] = 255
    #image_out *= mask
    sitk_img = sitk.GetImageFromArray(image_out.astype(np.uint8))
    print "filter finished in", timeit.default_timer() - stime, "s"
    sitk.WriteImage(sitk_img, os.path.join("./", imageout))


def bigaussian_filter3d(imagein, imageout, sigma_foreground=1.0, sigma_background=0.4, step_size=0.2, number_steps=3):
    img3d = sitk.GetArrayFromImage(sitk.ReadImage(imagein)).astype(np.float)
    max_value = np.amax(img3d)
    img3d *= (255.0 / max_value)
    stime = timeit.default_timer()
    image_out = np.zeros_like(img3d, dtype=np.float)
    return_list = list()
    p = sigma_background / sigma_foreground
    for i in range(number_steps):
        multiscale3DBG_step(img3d, sigma_foreground + (i * step_size), (sigma_foreground + (i * step_size)) * p, i, step_size, return_list)
    for result in return_list:
        image_out = np.maximum(image_out, result)

    image_out = np.clip(image_out, 0, 255)
    max_value = np.amax(image_out)
    if max_value < 255:
        image_out *= (255.0 / max_value)
    histogram = np.histogram(image_out, 255)
    threshold = max_entropy(histogram[0])

    sitk_img = sitk.GetImageFromArray(image_out.astype(np.uint8))
    print "filter finished in", timeit.default_timer() - stime, "s"
    sitk.WriteImage(sitk_img, os.path.join("./", imageout))

    mask = image_out > threshold
    image_out[mask] = 255
    image_out *= mask
    sitk_img = sitk.GetImageFromArray(image_out.astype(np.uint8))
    sitk.WriteImage(sitk_img, os.path.join("./", imageout+"threshold.mha"))


def bigaussian_filter2d(imagein, imageout, sigma_foreground=1.0, sigma_background=0.4, step_size=0.2, number_steps=3):
    array2d = sitk.GetArrayFromImage(sitk.ReadImage(imagein)).astype(np.float)
    return_list = list()
    p = sigma_background/sigma_foreground
    # convert to grayscale:
    if len(array2d.shape) == 3:
        array2d = np.mean(array2d, -1)
    image_out = np.zeros_like(array2d)
    stime = timeit.default_timer()
    for i in range(number_steps):
        multiscale2DBG_step(array2d, sigma_foreground + (i * step_size), (sigma_foreground + (i * step_size)) * p, i, step_size, return_list)
    for result in return_list:
        image_out = np.maximum(image_out, result)
    image_out = np.clip(image_out, 0, 255)
    max_value = np.amax(image_out)
    if max_value < 255:
        image_out *= (255.0 / max_value)
    threshold = rosin_threshold(image_out)
    mask = image_out > threshold
    #image_out[mask] = 255
    #image_out *= mask
    sitk_img2d = sitk.GetImageFromArray(image_out.astype(np.uint8))
    print "filter finished in", timeit.default_timer() - stime, "s"
    sitk.WriteImage(sitk_img2d, os.path.join("./", imageout))




def parallel_filter3d(imagein, imageout, sigma_foreground=1.0, sigma_background=0.4, step_size=0.2, number_steps=3):
    """Loads a 3D image, applies the filter in parallel, saves the result"""
    manager = multiprocessing.Manager()
    return_list = manager.list()
    jobs = []

    img3d = sitk.GetArrayFromImage(sitk.ReadImage(imagein)).astype(np.float)
    max_value = np.amax(img3d)
    img3d *= (255.0 / max_value)
    p = sigma_background / sigma_foreground

    stime = timeit.default_timer()
    for i in range(number_steps):
        proc = multiprocessing.Process(target=multiscale3DBG_step,
                                       args=(img3d, sigma_foreground + (i * step_size), (sigma_foreground + (i * step_size))*p, i, step_size, return_list))
        jobs.append(proc)
        proc.start()

    for proc in jobs:
        proc.join()
    image_out = np.zeros_like(img3d, dtype=np.float)
    for result in return_list:
        image_out = np.maximum(image_out, result)
    image_out = np.clip(image_out, 0, 255)
    max_value = np.amax(image_out)
    if max_value < 255:
        image_out *= (255.0 / max_value)
    sitk_img = sitk.GetImageFromArray(image_out.astype(np.uint8))
    print "parallel filter finished in", timeit.default_timer() - stime, "s"
    sitk.WriteImage(sitk_img, os.path.join("./", imageout))


def parallel_filter2d(imagein, imageout, sigma_foreground=1.0, sigma_background=0.4, step_size=0.2, number_steps=3):
    """Loads a 2D image, applies the filter in parallel, saves the result"""
    manager = multiprocessing.Manager()
    return_list = manager.list()
    array2d = sitk.GetArrayFromImage(sitk.ReadImage(imagein)).astype(np.float)
    # convert to grayscale
    if len(array2d.shape) == 3:
        array2d = np.mean(array2d, -1)
    image_out = np.zeros_like(array2d)
    p = sigma_background / sigma_foreground
    jobs = []
    stime = timeit.default_timer()
    for i in range(number_steps):
        proc = multiprocessing.Process(target=multiscale2DBG_step,
                                       args=(array2d, sigma_foreground + (i * step_size), (sigma_foreground + (i * step_size))*p, i, step_size, return_list))
        jobs.append(proc)
        proc.start()
    for proc in jobs:
        proc.join()
    for result in return_list:
        image_out = np.maximum(image_out, result)
    image_out = np.clip(image_out, 0, 255)
    max_value = np.amax(image_out)
    if max_value < 255:
        image_out *= (255.0 / max_value)
    sitk_img2d = sitk.GetImageFromArray(image_out.astype(np.uint8))
    print "parallel filter finished in", timeit.default_timer() - stime, "s"
    sitk.WriteImage(sitk_img2d, os.path.join("./", imageout))


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