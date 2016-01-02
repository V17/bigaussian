__author__ = 'Vojtech Vozab'
import numpy as np
import cv2
import SimpleITK as sitk
from scipy import ndimage
from scipy import signal
from scipy import sparse
import os
import timeit



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

    kernel = cv2.getGaussianKernel(ksize, sigma) + c0
    kernelb = cv2.getGaussianKernel(ksize, sigmab) * k
    kernel[0 : ((ksize-1) / 2) - sigma] = kernelb[sigma-sigmab : ((ksize-1) / 2) - sigmab]
    kernel[((ksize-1) / 2) + sigma : ksize] = kernelb[((ksize-1) / 2) + sigmab : ksize + sigmab - sigma]
    kernel_sum = sum(kernel)
    kernel_normalized = kernel / kernel_sum

    return kernel_normalized

def biGaussianKernel2D(sigma, sigmab, ksize=0):
    '''Generates a 2D bigaussian kernel from a 1D kernel using polar coordinates'''

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
    kernelf = cv2.getGaussianKernel(ksize, sigma) + c0
    kernelb = cv2.getGaussianKernel(ksize*2, sigmab) * k
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
    '''Generates a 3D bigaussian kernel from a 1D kernel using spherical coordinates'''

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
    kernelf = cv2.getGaussianKernel(ksize, sigma) + c0
    kernelb = cv2.getGaussianKernel(ksize*2, sigmab) * k
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
    '''Returns a hessian matrix for each pixel in a 2D image'''

    #kernels
    d2xx = np.array([[1, -2, 1]])
    d2yy = np.array([[1], [-2], [1]])
    d2xy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) * 0.25

    #magnitudes
    imagexx = cv2.filter2D(image, cv2.CV_32F, d2xx)
    #print "imagexx max, min", np.amax(imagexx), np.amin(imagexx)
    imagexy = cv2.filter2D(image, cv2.CV_32F, d2xy)
    imageyy = cv2.filter2D(image, cv2.CV_32F, d2yy)

    #imagexx = signal.fftconvolve(image, d2xx, mode="same")
    #imageyy = signal.fftconvolve(image, d2yy, mode="same")
    #imagexy = signal.fftconvolve(image, d2xy, mode="same")

    hessian = np.empty([2, 2, image.shape[0], image.shape[1]])
    hessian[0][0][:][:] = imagexx
    hessian[0][1][:][:] = imagexy
    hessian[1][0][:][:] = imagexy
    hessian[1][1][:][:] = imageyy

    return np.transpose(hessian, (2, 3, 0, 1))

def hessian3D(image):
    '''Returns a hessian matrix for each pixel in a 3D image'''

    #kernels
    d2xx = np.array([[[1, -2, 1]]])
    d2yy = np.array([[[1], [-2], [1]]])
    d2zz = np.array([[[1]], [[-2]], [[1]]])
    d2xy = np.array([[[-0.25, 0, 0.25], [0, 0, 0], [0.25, 0, -0.25]]])
    d2xz = np.array([[[-0.25, 0, 0.25]], [[0, 0, 0]], [[0.25, 0, -0.25]]])
    d2yz = np.array([[[-0.25], [0], [0.25]], [[0], [0], [0]], [[-0.25], [0], [0.25]]])

    #magnitudes
    imagexx = signal.fftconvolve(image, d2xx, mode="same")
    imageyy = signal.fftconvolve(image, d2yy, mode="same")
    imagezz = signal.fftconvolve(image, d2zz, mode="same")
    imagexy = signal.fftconvolve(image, d2xy, mode="same")
    imagexz = signal.fftconvolve(image, d2xz, mode="same")
    imageyz = signal.fftconvolve(image, d2yz, mode="same")

    hessian = np.empty([3, 3, image.shape[0], image.shape[1], image.shape[2]])
    hessian[0][0][:][:][:] = imagexx
    hessian[0][1][:][:][:] = imagexy
    hessian[0][2][:][:][:] = imagexz
    hessian[1][0][:][:][:] = imagexy
    hessian[1][1][:][:][:] = imageyy
    hessian[1][2][:][:][:] = imageyz
    hessian[2][0][:][:][:] = imagexz
    hessian[2][1][:][:][:] = imageyz
    hessian[2][2][:][:][:] = imagezz

    hessianmatrix = np.transpose(hessian, (2, 3, 4, 0, 1))

    return hessianmatrix

def eigenvalues3D(hessian):
    '''Returns eigenvalues of a 3x3 hessian of each pixel'''

    eigenvalues = np.zeros([hessian.shape[0], hessian.shape[1], hessian.shape[2], 3], float)
    eigenvalues[:][:][:] = np.linalg.eigvals(hessian[:][:][:])

    return eigenvalues

def lineness3D(eigenvalues):
    '''Returns lineness function value for each pixel computed from hessian eigenvalues'''

    result = np.zeros([eigenvalues.shape[0], eigenvalues.shape[1], eigenvalues.shape[2]])
    absmin = np.amax(np.fabs(eigenvalues), axis=3)
    absmax = np.amax(np.fabs(eigenvalues), axis=3)
    minval = np.amin(eigenvalues, axis=3)
    lambda1 = np.multiply(minval, np.equal(minval * (-1), absmax).astype(int))


    for z in range(eigenvalues.shape[0]):
        for y in range(eigenvalues.shape[1]):
            for x in range(eigenvalues.shape[2]):
                eigenvalues[z][y][x] = sorted(eigenvalues[z][y][x], key=abs)
                if np.sum(eigenvalues[z][y][x]) < 0:
                    result[z][y][x] = (-1) * (eigenvalues[z][y][x][1] / eigenvalues[z][y][x][2]) * (eigenvalues[z][y][x][1] + [eigenvalues[z][y][x][2]])

    return result

def max_eig2D(hessian):
    eigenvalues = np.zeros([hessian.shape[0], hessian.shape[1], 2], np.float32)
    eigenvalues[:][:] = np.linalg.eigvals(hessian[:][:])

    absmax = np.amax(np.fabs(eigenvalues), axis=2)
    minval = np.amin(eigenvalues, axis=2)
    result = np.multiply(minval, np.equal(minval * (-1), absmax))
    return result * (-1)

def max_eig2D_alt(hessian): #funguje taky, je potencialne srozumitelnejsi ale o kousek pomalejsi
    eigenvalues = np.zeros([hessian.shape[0], hessian.shape[1], 2], np.float32)
    eigenvalues[:][:] = np.linalg.eigvals(hessian[:][:])
    sorted_index = np.argsort(np.fabs(eigenvalues), axis=2)
    static_index = np.indices((hessian.shape[0], hessian.shape[1], 2))

    eigenvalues = eigenvalues[static_index[0], static_index[1], sorted_index] # tohle nefunguje protoze to je advanced indexing. Jak funguje advanced indexing???

    return (np.transpose(eigenvalues, (2, 0, 1))[1] * (-1)).clip(0)

def lineness3D_alt(eigenvalues):
    sorted_index = np.argsort(np.fabs(eigenvalues), axis=3)
    static_index = np.indices((eigenvalues.shape[0], eigenvalues.shape[1], eigenvalues.shape[2], 3))
    eigenvalues = np.transpose(eigenvalues[static_index[0], static_index[1], static_index[2], sorted_index], (3, 0, 1, 2))
    eigensum = np.sum(eigenvalues, axis=0)
    result = np.multiply(np.divide(eigenvalues[1], eigenvalues[2]), np.add(eigenvalues[1], eigenvalues[2])) * (-1)

    eigensum[eigensum >= 0] = 0
    eigensum[eigensum < 0] = 1
    result = np.multiply(result, eigensum)

    return result

def lineness3D_alt_old(eigenvalues):
    l1 = np.transpose(eigenvalues, (3, 0, 1, 2))[0] #TODO nepocitat pro kazdy krok, bacha, soucet musi bit mensi nez 0, nikoliv jen prvni hodnota
    l2 = np.transpose(eigenvalues, (3, 0, 1, 2))[1]
    l3 = np.transpose(eigenvalues, (3, 0, 1, 2))[2]

    absmax = np.amax(np.fabs(eigenvalues), axis=3)
    absmin = np.amin(np.fabs(eigenvalues), axis=3)
    minval = np.amin(eigenvalues, axis=3)

    print "eigenvalues", eigenvalues.shape, "lx", l1.shape, "absmax", absmax.shape, "absmin", absmin.shape

    l1 = np.multiply(l1, np.logical_not(np.equal(np.fabs(l1), absmax).astype(int)))
    print l1[0]
    l1 = np.multiply(l1, np.logical_not(np.equal(np.fabs(l1), absmin).astype(int)))
    l2 = np.multiply(l2, np.logical_not(np.equal(np.fabs(l2), absmax).astype(int)))
    l2 = np.multiply(l2, np.logical_not(np.equal(np.fabs(l2), absmin).astype(int)))
    l3 = np.multiply(l3, np.logical_not(np.equal(np.fabs(l3), absmax).astype(int)))
    l3 = np.multiply(l3, np.logical_not(np.equal(np.fabs(l3), absmin).astype(int)))
    med = np.add(np.add(l1, l2), l3)
    print "med", med.shape

    maxneg = np.multiply(minval, np.equal(minval * (-1), absmax).astype(int))
    maxneg = np.add(np.multiply(np.ones_like(maxneg), np.logical_not(np.equal(minval * (-1), absmax).astype(int))) * (-1), maxneg)
    result = np.multiply(np.divide(med, maxneg), np.add(med, maxneg)) * (-1)


    print np.amax(med), np.amin(med)


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

        print "bigaussian kernel generated in", timeit.default_timer() - stime
        stime = timeit.default_timer()

        img_filtered = cv2.filter2D(image, -1, kernel)

        print "image filtered in", timeit.default_timer() - stime
        stime = timeit.default_timer()

        img_hessian = hessian2D(img_filtered) #compute the hessian

        print "hessian computed in", timeit.default_timer() - stime
        stime = timeit.default_timer()

        #img_e = eigenvalues2D(img_hessian) #compute the eigenvalues from hessian
        img_e = max_eig2D_alt(img_hessian)

        print "eigenvalues and lineness computed in", timeit.default_timer() - stime


        #print "eigenvalues max, min", np.amax(img_e), np.amin(img_e)
        image_out = np.maximum(image_out, img_e) #compare to output image and take the higher intensity pixels

    max = np.amax(image_out)
    #print "maximum image_out pred normalizaci", max
    image_out *= (255.0 / max)
    #print "maximum image_out po normalizaci", np.amax(image_out)
    return image_out.astype(np.uint8) #normalize the image to 0-255 and return

def multiscale3DBG(image, sigmaf, sigmab, step, nsteps):
    '''Implements multiscale filtering for 3D images: for each step the image is blurred using accordingly sized
    bigaussian, hessian matrix is computed for each pixel and the largest (absolute) eigenvalue is found. If the
    eigenvalue intensity is larger than the intensity in the output image (initialized with zeros), it is used
    as the output value for that pixel.'''

    image_out = np.zeros_like(image, dtype=np.float16)
    for i in range(nsteps):

        stime = timeit.default_timer()

        kernel = biGaussianKernel3D(sigmaf + (i * step), sigmab + (i * step / 2))

        print "bigaussian kernel generated in", timeit.default_timer() - stime
        stime = timeit.default_timer()

        img_filtered = signal.fftconvolve(image, kernel, mode="same")

        print "image filtered in", timeit.default_timer() - stime
        stime = timeit.default_timer()

        img_hessian = hessian3D(img_filtered)

        print "hessian computed in", timeit.default_timer() - stime
        stime = timeit.default_timer()

        img_eigenvalues = eigenvalues3D(img_hessian).astype(np.float16)

        print "eigenvalues computed in", timeit.default_timer() - stime
        stime = timeit.default_timer()

        img_lineness = lineness3D_alt(img_eigenvalues).astype(np.float16)

        print "lineness filter response computed in", timeit.default_timer() - stime

        image_out = np.maximum(image_out, img_lineness)

    max = np.amax(image_out)
    return ((image_out/max)*255).astype(np.uint8)


################################################
# 2D filtrovani
#
#img = cv2.imread('gafa.jpg', 0)
#dst = multiscale2DBG(img, 0.8, 0.3, 0.1, 3)
#cv2.imwrite("gafa_alt.jpg", dst)
#
#3D filtrovani
#
img3d = sitk.ReadImage('Normal001-MRA.mha')
array3d = sitk.GetArrayFromImage(img3d)
dst = multiscale3DBG(array3d, 1, 0.4, 0.2, 3)
sitk_img = sitk.GetImageFromArray(dst)
sitk.WriteImage(sitk_img, os.path.join("./", 'normal001/test.mha'))

# PROBLEMY:
#
# - konvoluce pres fft pouzita u 3D obrazu ma problem s krajnimi pixely, u snimku mozku neni problem,
#   ale jinde byt muze - oriznout, normalizovat oriznute
#
# - fft s vetsimi nez malymi 3D obrazy pada, ale genericka konvoluce je straslive pomala
