# -*- coding: utf-8 -*-
__author__ = 'Vojtech Vozab'
import numpy as np
import SimpleITK as sitk
from scipy import ndimage, signal
import os
import timeit


def gaussian_kernel_1d(n, sigma):
    """Generates a 1D gaussian kernel using a built-in filter and a dirac impulse"""
    dirac = np.zeros(n)
    dirac[(n-1)/2] = 1
    return ndimage.filters.gaussian_filter1d(dirac, sigma)


def gaussian_kernel_2d(sigma, sigma_b, ksize=0):
    if ksize == 0:
        ksize = np.round(6 * sigma - 1)
    if (ksize % 2) == 0:
        ksize += 1
    dirac = np.zeros((ksize, ksize))
    dirac[(ksize-1)/2, (ksize-1)/2] = 1
    return ndimage.filters.gaussian_filter(dirac, sigma, mode='nearest')


def gaussian_kernel_3d(sigma, sigma_b, ksize=0):
    if ksize == 0:
        ksize = np.round(6 * sigma - 1)
    if (ksize % 2) == 0:
        ksize += 1
    dirac = np.zeros((ksize, ksize, ksize))
    dirac[(ksize-1)/2, (ksize-1)/2, (ksize-1)/2] = 1
    return ndimage.filters.gaussian_filter(dirac, sigma, mode='nearest')


def bigaussian_kernel_1d(sigma, sigmab, ksize=0):
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

    kernel = gaussian_kernel_1d(ksize, sigma) + c0
    kernelb = gaussian_kernel_1d(ksize, sigmab) * k

    kernel[0 : ((ksize-1) / 2) - sigma] = kernelb[sigma-sigmab : ((ksize-1) / 2) - sigmab]
    kernel[((ksize-1) / 2) + sigma : ksize] = kernelb[((ksize-1) / 2) + sigmab : ksize + sigmab - sigma]
    kernel_sum = sum(kernel)
    kernel_normalized = kernel / kernel_sum

    return kernel_normalized


def bigaussian_kernel_2d(sigma, sigmab, ksize=0):
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

    kernelf = np.asarray(gaussian_kernel_1d(ksize, sigma)) + c0
    kernelb = np.asarray(gaussian_kernel_1d(ksize*2, sigmab)) * k

    kernel2d = np.zeros([ksize, ksize], dtype=np.float)

    for y in range((ksize+1)/2-1, ksize-1):
        for x in range((ksize+1)/2-1, ksize-1):
            r = np.int(np.floor(np.sqrt((x-((ksize-1)/2))**2 + (y-((ksize-1)/2))**2)))  # distance from the center point
            u = r + ((ksize*2-1)/2)  # distance from the center point translated into indices for the 1D kernels
            v = r + ((ksize-1)/2)
            if r >= sigma:
                kernel2d[y][x] = kernelb[u + sigmab - sigma]
                kernel2d[ksize-1-y][x] = kernelb[u + sigmab - sigma]
                kernel2d[y][ksize-1-x] = kernelb[u + sigmab - sigma]
                kernel2d[ksize-1-y][ksize-1-x] = kernelb[u + sigmab - sigma]
            else:
                kernel2d[y][x] = kernelf[v]
                kernel2d[ksize-1-y][x] = kernelf[v]
                kernel2d[y][ksize-1-x] = kernelf[v]
                kernel2d[ksize-1-y][ksize-1-x] = kernelf[v]

    kernel2d = kernel2d / np.sum(kernel2d)  # normalization
    return kernel2d


def bigaussian_kernel_3d(sigma, sigmab, ksize=0):
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

    kernelf = gaussian_kernel_1d(ksize, sigma) + c0
    kernelb = gaussian_kernel_1d(ksize*2, sigmab) * k
    kernel3d = np.zeros([ksize, ksize, ksize])

    for y in range((ksize+1)/2-1, ksize-1):
        for x in range((ksize+1)/2-1, ksize-1):
            for z in range((ksize+1)/2-1, ksize-1):
                r = np.int(np.floor(np.sqrt((x-((ksize-1)/2))**2 + (y-((ksize-1)/2))**2 + (z-((ksize-1)/2))**2)))  # distance from the center point
                u = r + ((ksize*2-1)/2)  # distance from the center point translated into indices for the 1D kernels
                v = r + ((ksize-1)/2)
                if r >= sigma:
                    kernel3d[z][y][x] = kernelb[u + sigmab - sigma]
                    kernel3d[z][ksize-1-y][x] = kernelb[u + sigmab - sigma]
                    kernel3d[z][y][ksize-1-x] = kernelb[u + sigmab - sigma]
                    kernel3d[z][ksize-1-y][ksize-1-x] = kernelb[u + sigmab - sigma]
                    kernel3d[ksize-1-z][y][x] = kernelb[u + sigmab - sigma]
                    kernel3d[ksize-1-z][ksize-1-y][x] = kernelb[u + sigmab - sigma]
                    kernel3d[ksize-1-z][y][ksize-1-x] = kernelb[u + sigmab - sigma]
                    kernel3d[ksize-1-z][ksize-1-y][ksize-1-x] = kernelb[u + sigmab - sigma]
                else:
                    kernel3d[z][y][x] = kernelf[v]
                    kernel3d[z][ksize-1-y][x] = kernelf[v]
                    kernel3d[z][y][ksize-1-x] = kernelf[v]
                    kernel3d[z][ksize-1-y][ksize-1-x] = kernelf[v]
                    kernel3d[ksize-1-z][y][x] = kernelf[v]
                    kernel3d[ksize-1-z][ksize-1-y][x] = kernelf[v]
                    kernel3d[ksize-1-z][y][ksize-1-x] = kernelf[v]
                    kernel3d[ksize-1-z][ksize-1-y][ksize-1-x] = kernelf[v]

    kernel3d = kernel3d / np.sum(kernel3d, dtype=np.float)  # normalization
    return kernel3d


def hessian2d(image, sigma):
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


def hessian3d(image, sigma):
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


def max_eigenvalue_magnitude_2d(hessian):
    """Returns the eigenvalue with the largest absolute value for each pixel, sets negative values to zero"""
    eigenvalues = np.linalg.eigvals(hessian)
    sorted_index = np.argsort(np.fabs(eigenvalues), axis=2)
    static_index = np.indices((hessian.shape[0], hessian.shape[1], 2))
    eigenvalues = eigenvalues[static_index[0], static_index[1], sorted_index]

    return (np.transpose(eigenvalues, (2, 0, 1))[1] * (-1)).clip(0)


def lineness_bg_3d(eigenvalues):
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


def lineness_frangi_3d(eigenvalues):
    eigenvalues_abs = np.fabs(eigenvalues)
    sorted_index = np.argsort(eigenvalues_abs, axis=3)
    static_index = np.indices((eigenvalues.shape[0], eigenvalues.shape[1], eigenvalues.shape[2], 3))
    eigenvalues = np.transpose(eigenvalues[static_index[0], static_index[1], static_index[2], sorted_index], (3, 0, 1, 2))
    eigenvalues_abs = np.transpose(np.sort(eigenvalues_abs, axis=3), (3, 0, 1, 2))
    ra = eigenvalues_abs[1]/eigenvalues_abs[2]
    rb = eigenvalues_abs[0]/np.sqrt(eigenvalues_abs[1]*eigenvalues_abs[2])
    s = np.sqrt(eigenvalues_abs[0]**2 + eigenvalues_abs[1]**2 + eigenvalues_abs[2]**2)
    c = np.amax(s)/2
    result = (1 - np.exp(-(ra**2)/0.5)) * np.exp(-(rb**2)/0.5) * (1 - np.exp(- (s**2)/(2*(c**2))))
    mask1 = eigenvalues[1] < 0
    mask2 = eigenvalues[2] < 0
    result = (result * mask1) * mask2
    return result


def lineness_sato_3d(eigenvalues):
    eigenvalues = np.transpose(np.sort(eigenvalues, axis=3), (3, 0, 1, 2))
    alpha = 0.5
    output_a = np.fabs(eigenvalues[1]) + eigenvalues[2]
    output_b = np.fabs(eigenvalues[1]) - (alpha * eigenvalues[2])
    mask_a = eigenvalues[2] <= 0
    mask_b = np.logical_and(np.logical_and(eigenvalues[1] < 0, eigenvalues[2] > 0), eigenvalues[2] < (np.fabs(eigenvalues[1])/alpha))
    output_a *= mask_a
    output_b *= mask_b
    return output_a + output_b


def multiscale2DBG_step(image, sigmaf, sigmab, i, step, return_dict):
    """Single iteration of 2D bigaussian filter, stores the output in the return_dict list"""
    stime = timeit.default_timer()
    kernel = bigaussian_kernel_2d(sigmaf + (i * step), sigmab + (i * step / 2))
    print i+1, "- bigaussian kernel generated in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_filtered = ndimage.convolve(image, kernel, mode='nearest')
    print i+1, "- image filtered with bi-gaussian in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_hessian = hessian2d(img_filtered, sigmaf)
    print i+1, "- hessian computed in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_e = max_eigenvalue_magnitude_2d(img_hessian)
    print i+1, "- eigenvalues and lineness computed in", timeit.default_timer() - stime, "s"

    return_dict.append(img_e)
    return


def rosin_threshold(image):
    histogram = ndimage.histogram(image, 0, 255, 255)
    max_hist_index = np.argmax(histogram)
    min_hist_array = np.array(np.nonzero(histogram))[-1]
    min_hist_index = min_hist_array[-1]
    if min_hist_index < 254:
        min_hist_index += 1
    p1 = np.array([max_hist_index, histogram[max_hist_index]])
    p2 = np.array([min_hist_index, histogram[min_hist_index]])
    best_idx = -1
    max_dist = -1
    for x0 in range(max_hist_index, min_hist_index):
        y0 = histogram[x0]
        a = p1 - p2
        b = np.array([x0, y0]) - p2
        cross_ab = a[0] * b[1] - b[0] * a[1]
        d = np.linalg.norm(cross_ab) / np.linalg.norm(a)
        if d > max_dist:
            best_idx = x0
            max_dist = d
    print "threshold:", best_idx
    return best_idx


def max_entropy_threshold(histogram):
    """
    Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method
    Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for Gray-Level Picture Thresholding Using the Entropy
    of the Histogram", Graphical Models and Image Processing, 29(3): 273-285
    M. Emre Celebi
    06.15.2007
    Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
    2016-04-28: Adapted for Python 2.7 by Robert Metchev from Java source of MaxEntropy() in the Autothresholder plugin
    http://rsb.info.nih.gov/ij/plugins/download/AutoThresholder.java
    :param histogram: Sequence representing the histogram of the image
    :return threshold: Resulting maximum entropy threshold
    """

    # calculate CDF (cumulative density function)
    cdf = histogram.astype(np.float).cumsum()

    # find histogram's nonzero area
    valid_idx = np.nonzero(histogram)[0]
    first_bin = valid_idx[0]
    last_bin = valid_idx[-1]

    # initialize search for maximum
    max_ent, threshold = 0, 0

    for it in range(first_bin, last_bin + 1):
        # Background (dark)
        hist_range = histogram[:it + 1]
        hist_range = hist_range[hist_range != 0] / cdf[it]  # normalize within selected range & remove all 0 elements
        tot_ent = -np.sum(hist_range * np.log(hist_range))  # background entropy

        # Foreground/Object (bright)
        hist_range = histogram[it + 1:]
        # normalize within selected range & remove all 0 elements
        hist_range = hist_range[hist_range != 0] / (cdf[last_bin] - cdf[it])
        tot_ent -= np.sum(hist_range * np.log(hist_range))  # accumulate object entropy

        # find max
        if tot_ent > max_ent:
            max_ent, threshold = tot_ent, it

    return threshold


def filter_3d_step(image, kernel, i, sigma, return_dict, lineness):
    stime = timeit.default_timer()
    img_resized = np.pad(image, int((kernel.shape[0]/2)), mode='reflect')
    img_filtered = signal.fftconvolve(img_resized, kernel, mode='valid')
    print i+1, "- image smoothed in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_hessian = hessian3d(img_filtered, sigma)
    print i+1, "- hessian computed in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_eigenvalues = np.linalg.eigvals(img_hessian).astype(np.float)
    print i+1, "- eigenvalues computed in", timeit.default_timer() - stime, "s"

    stime = timeit.default_timer()
    img_lineness = lineness(img_eigenvalues).astype(np.float)
    print i+1, "- lineness filter response computed in", timeit.default_timer() - stime, "s"

    return_dict.append(img_lineness)
    return


def general_filter_3d(imagein, imageout, kernel_function, vesselness_function, sigma_foreground=3, sigma_background=1.5, step_size=0.5, number_steps=1):
    img3d = sitk.GetArrayFromImage(sitk.ReadImage(imagein)).astype(np.float)
    return_list = list()
    p = float(sigma_background)/float(sigma_foreground)
    image_out = np.zeros_like(img3d, dtype=np.float)
    stime = timeit.default_timer()
    for i in range(number_steps):
        kernel = kernel_function(sigma_foreground + (i * step_size), (sigma_foreground + (i * step_size)) * p)
        filter_3d_step(img3d, kernel, i,  sigma_foreground + (i * step_size), return_list, vesselness_function)
    for result in return_list:
        image_out = np.maximum(image_out, result)

    image_out = np.clip(image_out, 0, 255)
    max_value = np.amax(image_out)
    if max_value < 255:
        image_out *= (255.0 / max_value)

    sitk_img = sitk.GetImageFromArray(image_out.astype(np.uint8))
    print "filter finished in", timeit.default_timer() - stime, "s"
    filename, suffix = os.path.splitext(imageout)
    sitk.WriteImage(sitk_img, filename+"_out"+suffix)

    histogram = np.histogram(image_out, 255)[0]
    threshold = max_entropy_threshold(histogram)
    mask = image_out > threshold
    image_out[mask] = 255
    image_out *= mask
    sitk_img = sitk.GetImageFromArray(image_out.astype(np.uint8))
    sitk.WriteImage(sitk_img, os.path.join("./", filename+"_"+"out"+"_threshold"+suffix))
