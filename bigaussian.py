__author__ = 'Vojtech Vozab'
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import os
import timeit
import multiprocessing


def getGaussianKernel1d(n, sigma):
    '''Generates a 1D gaussian kernel using a built-in filter and a dirac impulse'''
    dirac = np.zeros(n)
    dirac[(n-1)/2] = 1
    return ndimage.filters.gaussian_filter1d(dirac, sigma)


def biGaussianKernel1D(sigma, sigmab, ksize=0):
    '''Generates a 1D bigaussian kernel'''

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
    '''Generates a 2D bigaussian kernel from 1D gaussian kernels using polar coordinates'''

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

    kernel2D = np.zeros([ksize, ksize])

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

    kernel2D = kernel2D / np.sum(kernel2D, dtype = np.float) #normalization
    return kernel2D


def biGaussianKernel3D(sigma, sigmab, ksize=0):
    '''Generates a 3D bigaussian kernel from 1D gaussian kernels using spherical coordinates'''

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

    kernel3D = kernel3D / np.sum(kernel3D, dtype = np.float) #normalization
    return kernel3D


def hessian2D(image):
    '''Returns a matrix of hessian matrices for each pixel in a 2D image'''

    #kernels
    d2xx = np.array([[1, -2, 1]])
    d2yy = np.array([[1], [-2], [1]])
    d2xy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) * 0.25

    #magnitudes
    imagexx = ndimage.filters.convolve(image, d2xx)
    imagexy = ndimage.filters.convolve(image, d2xy)
    imageyy = ndimage.filters.convolve(image, d2yy)

    hessian = np.empty([2, 2, image.shape[0], image.shape[1]])
    hessian[0][0][:][:] = imagexx
    hessian[0][1][:][:] = imagexy
    hessian[1][0][:][:] = imagexy
    hessian[1][1][:][:] = imageyy

    return np.transpose(hessian, (2, 3, 0, 1))


def hessian3D(image):
    '''Returns a matrix of hessian matrices for each pixel in a 3D image'''

    #kernels
    d2xx = np.array([[[1, -2, 1]]])
    d2yy = np.array([[[1], [-2], [1]]])
    d2zz = np.array([[[1]], [[-2]], [[1]]])
    d2xy = np.array([[[-0.25, 0, 0.25], [0, 0, 0], [0.25, 0, -0.25]]])
    d2xz = np.array([[[-0.25, 0, 0.25]], [[0, 0, 0]], [[0.25, 0, -0.25]]])
    d2yz = np.array([[[-0.25], [0], [0.25]], [[0], [0], [0]], [[-0.25], [0], [0.25]]])

    #magnitudes
    imagexx = ndimage.filters.convolve(image, d2xx)
    imageyy = ndimage.filters.convolve(image, d2yy)
    imagezz = ndimage.filters.convolve(image, d2zz)
    imagexy = ndimage.filters.convolve(image, d2xy)
    imagexz = ndimage.filters.convolve(image, d2xz)
    imageyz = ndimage.filters.convolve(image, d2yz)

    hessian = np.empty([3, 3, image.shape[0], image.shape[1], image.shape[2]], dtype=np.float16)
    hessian[0][0][:][:][:] = imagexx#.astype(np.float16)
    hessian[0][1][:][:][:] = imagexy#.astype(np.float16)
    hessian[0][2][:][:][:] = imagexz#.astype(np.float16)
    hessian[1][0][:][:][:] = imagexy#.astype(np.float16)
    hessian[1][1][:][:][:] = imageyy#.astype(np.float16)
    hessian[1][2][:][:][:] = imageyz#.astype(np.float16)
    hessian[2][0][:][:][:] = imagexz#.astype(np.float16)
    hessian[2][1][:][:][:] = imageyz#.astype(np.float16)
    hessian[2][2][:][:][:] = imagezz#.astype(np.float16)

    hessianmatrix = np.transpose(hessian, (2, 3, 4, 0, 1))

    return hessianmatrix


def eigenvalues3D(hessian):
    '''Returns a matrix of eigenvalues of a 3x3 hessian matrix of each pixel'''

    eigenvalues = np.linalg.eigvals(hessian.astype(np.float32))

    return eigenvalues


def max_eig2D_alt(hessian):
    '''Returns the eigenvalue with the largest absolute value for each pixel, sets negative values to zero'''
    eigenvalues = np.linalg.eigvals(hessian[:][:])
    sorted_index = np.argsort(np.fabs(eigenvalues), axis=2)
    static_index = np.indices((hessian.shape[0], hessian.shape[1], 2))

    eigenvalues = eigenvalues[static_index[0], static_index[1], sorted_index]
    return (np.transpose(eigenvalues, (2, 0, 1))[1] * (-1)).clip(0)


def max_eig2d_pool(hessian):
    pool = multiprocessing.Pool(processes = 4)
    eigenvalues = np.array(pool.map(np.linalg.eigvals, hessian))
    sorted_index = np.argsort(np.fabs(eigenvalues), axis=2)
    static_index = np.indices((hessian.shape[0], hessian.shape[1], 2))

    eigenvalues = eigenvalues[static_index[0], static_index[1], sorted_index]
    return (np.transpose(eigenvalues, (2, 0, 1))[1] * (-1)).clip(0)


def lineness3D(eigenvalues):
    '''Computes the 3D lineness function from eigenvalues for each pixel'''
    sorted_index = np.argsort(np.fabs(eigenvalues), axis=3)
    static_index = np.indices((eigenvalues.shape[0], eigenvalues.shape[1], eigenvalues.shape[2], 3))
    eigenvalues = np.transpose(eigenvalues[static_index[0], static_index[1], static_index[2], sorted_index], (3, 0, 1, 2))
    eigensum = np.sum(eigenvalues, axis=0)
    np.seterr(invalid='ignore') #function sometimes raises a division by zero warning, but this is ignored here and handled by nan_to_num() conversion
    result = np.multiply(np.nan_to_num(np.divide(eigenvalues[1], eigenvalues[2])), np.add(eigenvalues[1], eigenvalues[2])) * (-1)
    np.seterr(invalid='warn')

    eigensum[eigensum >= 0] = 0
    eigensum[eigensum < 0] = 1
    result = np.multiply(result, eigensum)

    return result


def multiscale2DBG(image, sigmaf, sigmab, step, nsteps):
    '''Implements multiscale filtering for 2D images: for each step the image is blurred using accordingly sized
    bigaussian, hessian matrix is computed for each pixel and the largest (absolute) eigenvalue is found. If the
    eigenvalue intensity is larger than the intensity in the output image (initialized with zeros), it is used
    as the output value for that pixel.'''

    max = np.amax(image)
    image = (image.astype(np.float32) / max) * 255
    image_out = np.zeros_like(image) #fill output image with zeros
    for i in range(nsteps):
        stime = timeit.default_timer()

        kernel = biGaussianKernel2D(sigmaf + (i * step), sigmab + (i * step / 2)) #generate the bigaussian kernel for each step

        print i+1, "- bigaussian kernel generated in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_filtered = ndimage.filters.convolve(image, kernel)

        print i+1, "- image filtered in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_hessian = hessian2D(img_filtered)

        print i+1, "- hessian computed in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_e = max_eig2D_alt(img_hessian)

        print i+1, "- eigenvalues and lineness computed in", timeit.default_timer() - stime, "s"

        image_out = np.maximum(image_out, img_e) #compare to output image and take the higher intensity pixels

    max = np.amax(image_out)
    image_out *= (255.0 / max)

    return image_out.astype(np.uint8) #normalize the image to 0-255 and return


def multiscale2DBG_step(image, sigmaf, sigmab, i, step, return_dict):
    stime = timeit.default_timer()

    kernel = biGaussianKernel2D(sigmaf + (i * step), sigmab + (i * step / 2)) #generate the bigaussian kernel for each step

    print i+1, "- bigaussian kernel generated in", timeit.default_timer() - stime, "s"
    stime = timeit.default_timer()

    img_filtered = ndimage.filters.convolve(image, kernel)

    print i+1, "- image filtered in", timeit.default_timer() - stime, "s"
    stime = timeit.default_timer()

    img_hessian = hessian2D(img_filtered)

    print i+1, "- hessian computed in", timeit.default_timer() - stime, "s"
    stime = timeit.default_timer()

    img_e = max_eig2D_alt(img_hessian)

    print i+1, "- eigenvalues and lineness computed in", timeit.default_timer() - stime, "s"

    return_dict.append(img_e)
    return


def multiscale3DBG_step(image, sigmaf, sigmab, i, step, return_dict):
    stime = timeit.default_timer()
    kernel = biGaussianKernel3D(sigmaf + (i * step), sigmab + (i * step / 2))
    print i+1, "- bigaussian kernel generated in", timeit.default_timer() - stime, "s"
    stime = timeit.default_timer()

    img_filtered = ndimage.filters.convolve(image.astype(np.float32), kernel.astype(np.float32))
    print i+1, "- image filtered in", timeit.default_timer() - stime, "s"
    stime = timeit.default_timer()

    img_hessian = hessian3D(img_filtered)

    print i+1, "- hessian computed in", timeit.default_timer() - stime, "s"
    stime = timeit.default_timer()

    img_eigenvalues = eigenvalues3D(img_hessian).astype(np.float16)

    print i+1, "- eigenvalues computed in", timeit.default_timer() - stime, "s"
    stime = timeit.default_timer()

    img_lineness = lineness3D(img_eigenvalues).astype(np.float16)

    print i+1, "- lineness filter response computed in", timeit.default_timer() - stime, "s"

    return_dict.append(img_lineness)
    return


def multiscale3DBG(image, sigmaf, sigmab, step, nsteps):
    '''Implements multiscale filtering for 3D images: for each step the image is blurred using accordingly sized
    bigaussian, hessian matrix is computed for each pixel and the lineness function is computed from its eigenvalues.
    If the lineness intensity is larger than the intensity in the output image (initialized with zeros), it is used
    as the output value for that pixel.'''

    image_out = np.zeros_like(image, dtype=np.float16)
    for i in range(nsteps):

        stime = timeit.default_timer()

        kernel = biGaussianKernel3D(sigmaf + (i * step), sigmab + (i * step / 2))

        print i+1, "- bigaussian kernel generated in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_filtered = ndimage.filters.convolve(image.astype(np.float32), kernel.astype(np.float32))

        print i+1, "- image filtered in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_hessian = hessian3D(img_filtered)

        print i+1, "- hessian computed in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_eigenvalues = eigenvalues3D(img_hessian).astype(np.float16)

        print i+1, "- eigenvalues computed in", timeit.default_timer() - stime, "s"
        stime = timeit.default_timer()

        img_lineness = lineness3D(img_eigenvalues).astype(np.float16)

        print i+1, "- lineness filter response computed in", timeit.default_timer() - stime, "s"

        image_out = np.maximum(image_out, img_lineness)


    max_value = np.amax(image_out)
    return ((image_out/max_value)*255).astype(np.uint8)


def filter3d(imagein, imageout, sigma_foreground=1, sigma_background=0.4, step_size=0.2, number_steps=3):
    '''Loads a 3D image, applies the filter, saves the result'''
    img3d = sitk.GetArrayFromImage(sitk.ReadImage(imagein))
    stime = timeit.default_timer()
    dst = multiscale3DBG(img3d, sigma_foreground, sigma_background, step_size, number_steps)
    sitk_img = sitk.GetImageFromArray(dst)
    print "single-thread filter finished in", timeit.default_timer() - stime, "s"
    sitk.WriteImage(sitk_img, os.path.join("./", imageout))


def filter2d(imagein, imageout, sigma_foreground=1, sigma_background=0.4, step_size=0.2, number_steps=3):
    '''Loads a 2D image, applies the filter, saves the result'''
    img2d = sitk.ReadImage(imagein)
    array2d = sitk.GetArrayFromImage(img2d)
    if len(array2d.shape) == 3:
        array2d = np.mean(array2d, -1) #converts to grayscale

    stime = timeit.default_timer()
    dst = multiscale2DBG(array2d, sigma_foreground, sigma_background, step_size, number_steps)
    sitk_img2d = sitk.GetImageFromArray(dst)
    print "single-thread filter finished in", timeit.default_timer() - stime, "s"
    sitk.WriteImage(sitk_img2d, os.path.join("./", imageout))


def parallel_filter3d(imagein, imageout, sigma_foreground=1, sigma_background=0.4, step_size=0.2, number_steps=3):
    manager = multiprocessing.Manager()
    return_dict = manager.list()
    img3d = sitk.GetArrayFromImage(sitk.ReadImage(imagein))
    jobs = []
    stime = timeit.default_timer()
    for i in range(number_steps):
        p = multiprocessing.Process(target=multiscale3DBG_step, args=(img3d, sigma_foreground + (i * step_size), sigma_background + (i * step_size / 2), i, step_size, return_dict))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()
    image_out = np.zeros_like(img3d, dtype=np.float16)
    for result in return_dict:
        image_out = np.maximum(image_out, result)
    #image_out = return_dict[1]
    max_value = np.amax(image_out)
    image_out *= (255.0 / max_value)
    sitk_img = sitk.GetImageFromArray(image_out.astype(np.uint8))
    print "parallel filter finished in", timeit.default_timer() - stime, "s"
    sitk.WriteImage(sitk_img, os.path.join("./", imageout))


def parallel_filter2d(imagein, imageout, sigma_foreground=1, sigma_background=0.4, step_size=0.2, number_steps=3):
    manager = multiprocessing.Manager()
    return_dict = manager.list()
    array2d = sitk.GetArrayFromImage(sitk.ReadImage(imagein))
    if len(array2d.shape) == 3:
        array2d = np.mean(array2d, -1) #converts to grayscale
    max_value = np.amax(array2d)
    array2d = (array2d.astype(np.float32) / max_value) * 255
    image_out = np.zeros_like(array2d) #fill output image with zeros
    jobs = []
    stime = timeit.default_timer()
    for i in range(number_steps):
        p = multiprocessing.Process(target=multiscale2DBG_step, args=(array2d, sigma_foreground + (i * step_size), sigma_background + (i * step_size / 2), i, step_size, return_dict))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()
    for result in return_dict:
        image_out = np.maximum(image_out, result)
    max_value = np.amax(image_out)
    image_out *= (255.0 / max_value)
    sitk_img2d = sitk.GetImageFromArray(image_out.astype(np.uint8))
    print "parallel filter finished in", timeit.default_timer() - stime, "s"
    sitk.WriteImage(sitk_img2d, os.path.join("./", imageout))

