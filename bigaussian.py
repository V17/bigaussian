# -*- coding: utf-8 -*-
__author__ = 'Vojtech Vozab'
import numpy as np
import SimpleITK as sitk
from scipy import ndimage, signal
import os
import timeit
import multiprocessing


def getGaussianKernel1d(n, sigma):
    """Generates a 1D gaussian kernel using a built-in filter and a dirac impulse"""
    dirac = np.zeros(n)
    dirac[(n-1)/2] = 1
    return ndimage.filters.gaussian_filter1d(dirac, sigma)


def biGaussianKernel1D(sigma, sigmab, ksize=0):
    """Generates a 1D bigaussian kernel"""

    if ksize == 0:
        ksize = np.round(6 * sigma - 1)

    if sigmab > sigma:
        print("Invalid arguments, sigmab must be smaller than or equal to sigma")
        return -1

    if (ksize % 2) == 0:
        ksize += 1
        print("size of kernel is even, enlarging by one")

    c0 = (np.exp(-0.5) / np.sqrt(2*np.pi)) * ((float(sigmab) / float(sigma)) - 1) * (1 / float(sigma))
    k = (float(sigmab**2) / float(sigma**2))

    kernel = getGaussianKernel1d(ksize, sigma) + c0
    kernelb = getGaussianKernel1d(ksize, sigmab) * k

    kernel[0 : ((ksize-1) / 2) - sigma] = kernelb[sigma-sigmab : ((ksize-1) / 2) - sigmab]
    kernel[((ksize-1) / 2) + sigma : ksize] = kernelb[((ksize-1) / 2) + sigmab : ksize + sigmab - sigma]
    kernel_sum = sum(kernel)
    kernel_normalized = kernel / kernel_sum

    return kernel_normalized


def biGaussianKernel2D(sigma, sigmab, ksize=0):
    """Generates a 2D bigaussian kernel from 1D gaussian kernels using polar coordinates"""

    if (sigma <= 0) | (sigmab <= 0) | (ksize < 0):
        print "All arguments have to be larger than 0"
        return -1

    if ksize == 0:
        ksize = int(np.round(6 * sigma - 1))

    if sigmab > sigma:
        print("Invalid arguments, sigmab must be smaller than or equal to sigma")
        return -1

    if (ksize % 2) == 0:
        ksize += 1
        print("size of kernel is even, enlarging by one")

    c0 = (np.exp(-0.5) / np.sqrt(2*np.pi)) * ((float(sigmab) / float(sigma)) - 1) * (1 / float(sigma))
    k = (float(sigmab**2) / float(sigma**2))

    kernelf = np.asarray(getGaussianKernel1d(ksize, sigma)) + c0
    kernelb = np.asarray(getGaussianKernel1d(ksize*2, sigmab)) * k

    kernel2D = np.zeros([ksize, ksize], dtype=np.float)

    for y in range((ksize+1)/2-1, ksize-1):
        for x in range((ksize+1)/2-1, ksize-1):
            r = np.int(np.floor(np.sqrt((x-((ksize-1)/2))**2 + (y-((ksize-1)/2))**2))) #distance from the center point
            u = r + ((ksize*2-1)/2) #distance from the center point translated into indices for the 1D kernels
            v = r + ((ksize-1)/2)
            if r >= sigma:
                kernel2D[y][x] = kernelb[u + sigmab - sigma]
                kernel2D[ksize-1-y][x] = kernelb[u + sigmab - sigma]
                kernel2D[y][ksize-1-x] = kernelb[u + sigmab - sigma]
                kernel2D[ksize-1-y][ksize-1-x] = kernelb[u + sigmab - sigma]
            else:
                kernel2D[y][x] = kernelf[v]
                kernel2D[ksize-1-y][x] = kernelf[v]
                kernel2D[y][ksize-1-x] = kernelf[v]
                kernel2D[ksize-1-y][ksize-1-x] = kernelf[v]

    kernel2D = kernel2D / np.sum(kernel2D) #normalization
    return kernel2D


def biGaussianKernel3D(sigma, sigmab, ksize=0):
    """Generates a 3D bigaussian kernel from 1D gaussian kernels using spherical coordinates"""

    if (sigma <= 0) | (sigmab <= 0) | (ksize < 0):
        print "All arguments have to be larger than 0"
        return -1

    if ksize == 0:
        ksize = int(np.round(6 * sigma - 1))

    if sigmab > sigma:
        print("Invalid arguments, sigmab must be smaller than or equal to sigma")
        return -1

    if (ksize % 2) == 0:
        ksize += 1
        print("size of kernel is even, enlarging by one")

    c0 = (np.exp(-0.5) / np.sqrt(2*np.pi)) * ((float(sigmab) / float(sigma)) - 1) * (1 / float(sigma))
    k = (float(sigmab**2) / float(sigma**2))

    kernelf = getGaussianKernel1d(ksize, sigma) + c0
    kernelb = getGaussianKernel1d(ksize*2, sigmab) * k
    kernel3D = np.zeros([ksize, ksize, ksize])

    for y in range((ksize+1)/2-1, ksize-1):
        for x in range((ksize+1)/2-1, ksize-1):
            for z in range((ksize+1)/2-1, ksize-1):
                r = np.int(np.floor(np.sqrt((x-((ksize-1)/2))**2 + (y-((ksize-1)/2))**2 + (z-((ksize-1)/2))**2))) #distance from the center point
                u = r + ((ksize*2-1)/2) #distance from the center point translated into indices for the 1D kernels
                v = r + ((ksize-1)/2)
                if r >= sigma:
                    kernel3D[z][y][x] = kernelb[u + sigmab - sigma]
                    kernel3D[z][ksize-1-y][x] = kernelb[u + sigmab - sigma]
                    kernel3D[z][y][ksize-1-x] = kernelb[u + sigmab - sigma]
                    kernel3D[z][ksize-1-y][ksize-1-x] = kernelb[u + sigmab - sigma]
                    kernel3D[ksize-1-z][y][x] = kernelb[u + sigmab - sigma]
                    kernel3D[ksize-1-z][ksize-1-y][x] = kernelb[u + sigmab - sigma]
                    kernel3D[ksize-1-z][y][ksize-1-x] = kernelb[u + sigmab - sigma]
                    kernel3D[ksize-1-z][ksize-1-y][ksize-1-x] = kernelb[u + sigmab - sigma]
                else:
                    kernel3D[z][y][x] = kernelf[v]
                    kernel3D[z][ksize-1-y][x] = kernelf[v]
                    kernel3D[z][y][ksize-1-x] = kernelf[v]
                    kernel3D[z][ksize-1-y][ksize-1-x] = kernelf[v]
                    kernel3D[ksize-1-z][y][x] = kernelf[v]
                    kernel3D[ksize-1-z][ksize-1-y][x] = kernelf[v]
                    kernel3D[ksize-1-z][y][ksize-1-x] = kernelf[v]
                    kernel3D[ksize-1-z][ksize-1-y][ksize-1-x] = kernelf[v]

    kernel3D = kernel3D / np.sum(kernel3D, dtype=np.float) #normalization
    return kernel3D


def hessian2D(image, sigma):
    """Returns a matrix of hessian matrices for each pixel in a 2D image"""
    [dy, dx] = np.gradient(image)
    [dyy, dyx] = np.gradient(dy)
    [dxy, dxx] = np.gradient(dx)

    hessian = np.empty([2, 2, image.shape[0], image.shape[1]])
    hessian[0, 0] = dxx * sigma**2
    hessian[0, 1] = dxy * sigma**2
    hessian[1, 0] = dyx * sigma**2
    hessian[1, 1] = dyy * sigma**2

    return np.transpose(hessian, (2, 3, 0, 1))


def hessian3D(image, sigma):
    """Returns a matrix of hessian matrices for each pixel in a 3D image"""

    [dz, dy, dx] = np.gradient(image)
    [dzz, dzy, dzx] = np.gradient(dz)
    [dyz, dyy, dyx] = np.gradient(dy)
    [dxz, dxy, dxx] = np.gradient(dx)

    hessian = np.empty([3, 3, image.shape[0], image.shape[1], image.shape[2]], dtype=np.float)
    hessian[0, 0] = dxx * sigma**2
    hessian[0, 1] = dxy * sigma**2
    hessian[0, 2] = dxz * sigma**2
    hessian[1, 0] = dxy * sigma**2
    hessian[1, 1] = dyy * sigma**2
    hessian[1, 2] = dyz * sigma**2
    hessian[2, 0] = dxz * sigma**2
    hessian[2, 1] = dyz * sigma**2
    hessian[2, 2] = dzz * sigma**2
    hessianmatrix = np.transpose(hessian, (2, 3, 4, 0, 1))

    return hessianmatrix


def eigenvalues3D(hessian):
    """Returns a matrix of eigenvalues of a 3x3 hessian matrix of each pixel"""

    eigenvalues = np.linalg.eigvals(hessian)

    return eigenvalues


def max_eig2D_alt(hessian):
    """Returns the eigenvalue with the largest absolute value for each pixel, sets negative values to zero"""
    eigenvalues = np.linalg.eigvals(hessian)
    sorted_index = np.argsort(np.fabs(eigenvalues), axis=2)
    static_index = np.indices((hessian.shape[0], hessian.shape[1], 2))
    eigenvalues = eigenvalues[static_index[0], static_index[1], sorted_index]
    return (np.transpose(eigenvalues, (2, 0, 1))[1] * (-1)).clip(0)


def lineness3D(eigenvalues):
    """Computes the 3D lineness function from eigenvalues for each pixel"""
    sorted_index = np.argsort(np.fabs(eigenvalues), axis=3)
    static_index = np.indices((eigenvalues.shape[0], eigenvalues.shape[1], eigenvalues.shape[2], 3))
    eigenvalues = np.transpose(eigenvalues[static_index[0], static_index[1], static_index[2], sorted_index], (3, 0, 1, 2))
    eigensum = np.sum(eigenvalues, axis=0)
    # function sometimes raises a division by zero warning, but this is ignored here and handled by nan_to_num() conversion
    np.seterr(invalid='ignore')
    result = np.multiply(np.nan_to_num(np.divide(eigenvalues[1], eigenvalues[2])), np.add(eigenvalues[1], eigenvalues[2])) * (-1)
    np.seterr(invalid='warn')

    eigensum[eigensum >= 0] = 0
    eigensum[eigensum < 0] = 1
    result = np.multiply(result, eigensum)

    return result


def multiscale2DBG_step(image, sigmaf, sigmab, i, step, return_dict):
    """Single iteration of 2D bigaussian filter, used for parallel filtering"""
    stime = timeit.default_timer()
    kernel = biGaussianKernel2D(sigmaf + (i * step), sigmab + (i * step / 2)) #generate the bigaussian kernel for each step
    print i+1, "- bigaussian kernel generated in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_filtered = signal.fftconvolve(image, kernel, mode='same')
    print i+1, "- image filtered with bi-gaussian in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_hessian = hessian2D(img_filtered, sigmaf)
    print i+1, "- hessian computed in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_e = max_eig2D_alt(img_hessian)
    print i+1, "- eigenvalues and lineness computed in", timeit.default_timer() - stime, "s"

    return_dict.append(img_e)
    return


def multiscale3DBG_step(image, sigmaf, sigmab, i, step, return_dict):
    """Single iteration of 3D bigaussian filter, used for parallel filtering"""
    stime = timeit.default_timer()
    kernel = biGaussianKernel3D(sigmaf + (i * step), sigmab + (i * step / 2))
    print i+1, "- bigaussian kernel generated in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_filtered = signal.fftconvolve(image, kernel, mode='same')
    print i+1, "- image filtered with bi-gaussian in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_hessian = hessian3D(img_filtered, sigmaf + (i * step))
    print i+1, "- hessian computed in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_eigenvalues = eigenvalues3D(img_hessian).astype(np.float16)
    print i+1, "- eigenvalues computed in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_lineness = lineness3D(img_eigenvalues).astype(np.float16)
    print i+1, "- lineness filter response computed in", timeit.default_timer() - stime, "s"

    return_dict.append(img_lineness)
    return


def filter3d(imagein, imageout, sigma_foreground=1.0, sigma_background=0.4, step_size=0.2, number_steps=3):
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
    sitk_img = sitk.GetImageFromArray(image_out.astype(np.uint8))
    print "filter finished in", timeit.default_timer() - stime, "s"
    sitk.WriteImage(sitk_img, os.path.join("./", imageout))


def filter2d(imagein, imageout, sigma_foreground=1.0, sigma_background=0.4, step_size=0.2, number_steps=3):
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
