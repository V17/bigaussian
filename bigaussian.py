# -*- coding: utf-8 -*-
# Copyright 2017 Vojtech Vozab
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


__author__ = 'Vojtech Vozab'
import numpy as np
from scipy import ndimage, signal
import timeit


def gaussian_1d_function(x, x0, sigma):
    return np.exp(-np.power((x - x0)/sigma, 2.)/2.)


def gaussian_kernel_1d(n, sigma):
    """Generates a 1D gaussian kernel using a built-in filter and a dirac impulse"""
    dirac = np.zeros(n)
    dirac[(n-1)/2] = 1
    return ndimage.filters.gaussian_filter1d(dirac, sigma)


def gaussian_kernel_2d(sigma, sigma_b=0, zratio=1, ksize=0):
    if ksize == 0:
        ksize = np.round(6 * sigma - 1)
    if (ksize % 2) == 0:
        ksize += 1
    xgauss = gaussian_kernel_1d(ksize, sigma*zratio)
    ygauss = gaussian_kernel_1d(ksize, sigma)
    kernel2d = np.outer(xgauss, ygauss)

    kernel2d = kernel2d / np.sum(kernel2d)
    return kernel2d


def gaussian_kernel_3d(sigma, sigma_b=0, zratio=1, ksize=0):
    if ksize == 0:
        ksize = np.round(6 * sigma - 1)
    if (ksize % 2) == 0:
        ksize += 1
    sigmaz = sigma * zratio
    if sigma*zratio < 1:
        sigmaz = 1
    ksize2 = np.round(6 * sigmaz - 1)
    if (ksize2 % 2) == 0:
        ksize2 += 1
    xgauss = gaussian_kernel_1d(ksize, sigma)
    ygauss = gaussian_kernel_1d(ksize, sigma)
    zgauss = gaussian_kernel_1d(ksize2, sigmaz)
    kernel3d = np.outer(zgauss, ygauss).reshape((ksize2, ksize, 1)) * xgauss

    kernel3d = kernel3d / np.sum(kernel3d)
    return kernel3d


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

    c0 = (np.exp(-0.5) / np.sqrt(2*np.pi)) * ((float(sigmab) / float(sigma)) - 1) * (1 / float(sigma))
    k = (float(sigmab**2) / float(sigma**2))

    kernel2d = np.zeros([ksize, ksize], dtype=np.float)

    for y in range((ksize+1)/2-1, ksize-1):
        for x in range((ksize+1)/2-1, ksize-1):
            r = np.sqrt((x-((ksize-1)/2.))**2 + (y-((ksize-1)/2.))**2)  # distance from the center point
            if r >= sigma:
                kernel2d[y][x] = gaussian_1d_function(r + sigmab - sigma, 0, sigmab) * k
                kernel2d[ksize-1-y][x] = gaussian_1d_function(r + sigmab - sigma, 0, sigmab) * k
                kernel2d[y][ksize-1-x] = gaussian_1d_function(r + sigmab - sigma, 0, sigmab) * k
                kernel2d[ksize-1-y][ksize-1-x] = gaussian_1d_function(r + sigmab - sigma, 0, sigmab) * k
            else:
                kernel2d[y][x] = gaussian_1d_function(r, 0, sigma) + c0
                kernel2d[ksize-1-y][x] = gaussian_1d_function(r, 0, sigma) + c0
                kernel2d[y][ksize-1-x] = gaussian_1d_function(r, 0, sigma) + c0
                kernel2d[ksize-1-y][ksize-1-x] = gaussian_1d_function(r, 0, sigma) + c0

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

    c0 = (np.exp(-0.5) / np.sqrt(2*np.pi)) * ((float(sigmab) / float(sigma)) - 1) * (1 / float(sigma))
    k = (float(sigmab**2) / float(sigma**2))

    kernel3d = np.zeros([ksize, ksize, ksize])

    for y in range((ksize+1)/2-1, ksize-1):
        for x in range((ksize+1)/2-1, ksize-1):
            for z in range((ksize+1)/2-1, ksize-1):
                r = np.int(np.floor(np.sqrt((x-((ksize-1)/2))**2 + (y-((ksize-1)/2))**2 + (z-((ksize-1)/2))**2)))  # distance from the center point

                if r >= sigma:
                    value = gaussian_1d_function(r + sigmab - sigma, 0, sigmab) * k
                    kernel3d[z][y][x] = value
                    kernel3d[z][ksize-1-y][x] = value
                    kernel3d[z][y][ksize-1-x] = value
                    kernel3d[z][ksize-1-y][ksize-1-x] = value
                    kernel3d[ksize-1-z][y][x] = value
                    kernel3d[ksize-1-z][ksize-1-y][x] = value
                    kernel3d[ksize-1-z][y][ksize-1-x] = value
                    kernel3d[ksize-1-z][ksize-1-y][ksize-1-x] = value
                else:
                    value = gaussian_1d_function(r, 0, sigma) + c0
                    kernel3d[z][y][x] = value
                    kernel3d[z][ksize-1-y][x] = value
                    kernel3d[z][y][ksize-1-x] = value
                    kernel3d[z][ksize-1-y][ksize-1-x] = value
                    kernel3d[ksize-1-z][y][x] = value
                    kernel3d[ksize-1-z][ksize-1-y][x] = value
                    kernel3d[ksize-1-z][y][ksize-1-x] = value
                    kernel3d[ksize-1-z][ksize-1-y][ksize-1-x] = value

    kernel3d = kernel3d / np.sum(kernel3d, dtype=np.float)  # normalization
    return kernel3d


def bigaussian_kernel_3d_alt(sigma, sigma_b, zratio=1, ksize=0):
    kernel = bigaussian_kernel_3d(sigma, sigma_b, ksize)
    #if zratio * sigma < 1:  # z size of kernel would be < 5 voxels
    #    print "z ratio too small, enlarging"
    #    zratio = 1 / sigma
    kernel_interp = ndimage.interpolation.zoom(kernel, (zratio, 1, 1), order=1)
    if kernel_interp.shape[0] < 5:
        kernel_interp = ndimage.interpolation.zoom(kernel, (1.0 / (kernel.shape[1] / 5.0), 1, 1), order=1)
        "z size < 5, enlarging to 5"
    if kernel_interp.shape[0] % 2 == 0:
        "z size even, enlarging by 1"
        kernel_interp = ndimage.interpolation.zoom(kernel, (zratio + (1.0/sigma/5.0), 1, 1), order=1)
    return kernel_interp / np.sum(kernel_interp)


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
    """Computes the bi-Gaussian 3D lineness function from eigenvalues for each pixel"""
    sorted_index = np.argsort(np.fabs(eigenvalues), axis=3)
    static_index = np.indices((eigenvalues.shape[0], eigenvalues.shape[1], eigenvalues.shape[2], 3))
    eigenvalues = np.transpose(eigenvalues[static_index[0], static_index[1], static_index[2], sorted_index], (3, 0, 1, 2))
    eigensum = np.sum(eigenvalues, axis=0)
    # function sometimes raises a division by zero warning, but this is ignored here and correctly handled by nan_to_num() conversion
    np.seterr(invalid='ignore')
    result = np.multiply(np.nan_to_num(np.divide(eigenvalues[1], eigenvalues[2])), np.add(eigenvalues[1], eigenvalues[2])) * (-1)
    np.seterr(invalid='warn')

    eigensum[eigensum >= 0] = 0
    eigensum[eigensum < 0] = 1
    result = np.multiply(result, eigensum)

    return result


def lineness_frangi_3d(eigenvalues):
    """Computes the Frangi 3D lineness function from eigenvalues for each pixel"""
    eigenvalues_abs = np.fabs(eigenvalues)
    sorted_index = np.argsort(eigenvalues_abs, axis=3)
    static_index = np.indices((eigenvalues.shape[0], eigenvalues.shape[1], eigenvalues.shape[2], 3))
    eigenvalues = np.transpose(eigenvalues[static_index[0], static_index[1], static_index[2], sorted_index], (3, 0, 1, 2))
    eigenvalues_abs = np.transpose(np.sort(eigenvalues_abs, axis=3), (3, 0, 1, 2))
    np.seterr(invalid='ignore')
    ra = np.nan_to_num(eigenvalues_abs[1]/eigenvalues_abs[2])
    rb = np.nan_to_num(eigenvalues_abs[0]/np.sqrt(eigenvalues_abs[1]*eigenvalues_abs[2]))
    np.seterr(invalid='warn')
    s = np.sqrt(eigenvalues_abs[0]**2 + eigenvalues_abs[1]**2 + eigenvalues_abs[2]**2)
    c = np.amax(s)/2
    result = (1 - np.exp(-(ra**2)/0.5)) * np.exp(-(rb**2)/0.5) * (1 - np.exp(- (s**2)/(2*(c**2))))
    mask1 = eigenvalues[1] < 0
    mask2 = eigenvalues[2] < 0
    result = (result * mask1) * mask2
    return result


def lineness_sato_3d(eigenvalues):
    """Computes the Sato 3D lineness function from eigenvalues for each pixel"""
    eigenvalues = np.transpose(np.sort(eigenvalues, axis=3), (3, 0, 1, 2))
    alpha = 0.5
    output_a = np.fabs(eigenvalues[1]) + eigenvalues[2]
    output_b = np.fabs(eigenvalues[1]) - (alpha * eigenvalues[2])
    mask_a = eigenvalues[2] <= 0
    mask_b = np.logical_and(np.logical_and(eigenvalues[1] < 0, eigenvalues[2] > 0), eigenvalues[2] < (np.fabs(eigenvalues[1])/alpha))
    output_a *= mask_a
    output_b *= mask_b
    return output_a + output_b


def filter_3d_step(image, kernel, i, sigma, return_dict, lineness):
    """Computes a single scale-step of a 3d filter on an image in the form of a numpy array. Accepts different kernels
    and lineness functions as arguments, stores the output image in return_dict."""
    xypad = int(kernel.shape[1] / 2)
    zpad = int(np.round(kernel.shape[0] / 2))
    print "kernel size:", kernel.shape
    img_resized = np.pad(image, [(zpad, zpad), (xypad, xypad), (xypad, xypad)], mode='reflect')
    img_filtered = signal.fftconvolve(img_resized, kernel, mode='valid')

    img_hessian = hessian3d(img_filtered, sigma)

    img_eigenvalues = np.linalg.eigvals(img_hessian).astype(np.float32)

    img_lineness = lineness(img_eigenvalues).astype(np.float32)
    print "max value for this step", np.max(img_lineness)

    return_dict[0] = np.maximum(return_dict[0], img_lineness)
    return


def general_filter_3d(img3d, kernel_function, vesselness_function, sigma_foreground=3, sigma_background=1.5, step_size=0.5, number_steps=1, zratio=1):
    """Applies a multi-scale filter on an image, enhances the contrast (if maximum intensity < 255), saves the output
    and then computes the threshold using max_entropy thresholding and saves the thresholded output.
    
    @imagein: numpy array of floats"""

    return_list = list()
    p = float(sigma_background)/float(sigma_foreground)
    image_out = np.zeros_like(img3d, dtype=np.float64)
    return_list.append(image_out)
    print "filter started"
    stime = timeit.default_timer()
    for i in range(number_steps):
        print "computing for sigma "+str(sigma_foreground + (i * step_size))
        kernel = kernel_function(sigma_foreground + (i * step_size), (sigma_foreground + (i * step_size)) * p, zratio)
        filter_3d_step(img3d, kernel, i, sigma_foreground + (i * step_size), return_list, vesselness_function)
    image_out = return_list[0]

    print "filter finished in", timeit.default_timer() - stime, "s"
    return image_out

    #histogram = np.histogram(image_out, 255)[0]
    #threshold = max_entropy_threshold.max_entropy_threshold(histogram)
    #mask = image_out > threshold
    #image_out[mask] = 255
    #image_out *= mask
    #sitk_img = sitk.GetImageFromArray(image_out.astype(np.uint8))
    #sitk.WriteImage(sitk_img, os.path.join("./", filename+"_"+"out"+"_threshold"+suffix))
    #print "output and thresholded output saved"
