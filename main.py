# -*- coding: utf-8 -*-
__author__ = 'Vojtech Vozab'
import bigaussian
import argparse
import glob
import numpy as np
import os
from skimage import io


def process_16bit_folder(path, kernel_function, vesselness_function, sigma_foreground, sigma_background, step_size, number_steps, zratio):
    suffix = 'tif'
    image_list = glob.glob(path + "*" + suffix)
    if not image_list:
        print "no loadable files in folder"
        return -1

    for image_file in image_list:
        process_16bit_file(image_file, None, bigaussian.bigaussian_kernel_3d_alt, bigaussian.lineness_bg_3d, sigma_foreground, sigma_background,
                           step_size, number_steps, zratio)
        process_16bit_file(image_file, None, bigaussian.bigaussian_kernel_3d_alt, bigaussian.lineness_frangi_3d, sigma_foreground, sigma_background,
                           step_size, number_steps, zratio)
        process_16bit_file(image_file, None, bigaussian.bigaussian_kernel_3d_alt, bigaussian.lineness_sato_3d, sigma_foreground, sigma_background,
                           step_size, number_steps, zratio)
        process_16bit_file(image_file, None, bigaussian.gaussian_kernel_3d_alt, bigaussian.lineness_bg_3d, sigma_foreground, sigma_background,
                           step_size, number_steps, zratio)
        process_16bit_file(image_file, None, bigaussian.gaussian_kernel_3d_alt, bigaussian.lineness_frangi_3d, sigma_foreground, sigma_background,
                           step_size, number_steps, zratio)
        process_16bit_file(image_file, None, bigaussian.gaussian_kernel_3d_alt, bigaussian.lineness_sato_3d, sigma_foreground, sigma_background,
                           step_size, number_steps, zratio)


def process_16bit_file(input_file, output_file, kernel_function, vesselness_function, sigma_foreground, sigma_background, step_size, number_steps, zratio):
    img_3d_float = io.imread(input_file).astype(np.float64) / 65535
    if kernel_function is bigaussian.bigaussian_kernel_3d_alt:
        dirname = "out_bg"
    else:
        dirname = "out_gauss"
    if vesselness_function is bigaussian.lineness_bg_3d:
        dirname = dirname+"_bg"
    elif vesselness_function is bigaussian.lineness_frangi_3d:
        dirname = dirname+"_frangi"
    else:
        dirname = dirname+"_sato"
    if output_file is None:
        directory, filename = os.path.split(input_file)
        filename_nosuf, suffix = os.path.splitext(filename)
        if not os.path.exists(os.path.join(directory, dirname)):
            os.makedirs(os.path.join(directory, dirname))
        print "processing", filename, "for", dirname
        output_file = os.path.join(directory, dirname, filename_nosuf)+"_out"+suffix
    print "filter params", sigma_foreground, sigma_background, step_size, number_steps
    output_3d_float = bigaussian.general_filter_3d(img_3d_float, kernel_function, vesselness_function, sigma_foreground, sigma_background, step_size, number_steps, zratio)
    io.imsave(output_file, (output_3d_float * 65535).astype(np.uint16))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A lineness filter for 3D images.')
    parser.add_argument('input', help='input filename')
    parser.add_argument('--output', '-o', help='output filename')
    parser.add_argument('--params', '-p', metavar='X', type=float, nargs=4, help='Filter parameters - foreground sigma and '
                                                                             'background sigma for bigaussian, number of '
                                                                             'multiscale steps and the value by which'
                                                                             'sigma gets enlarged each step, '
                                                                             'in this order. If omitted, the default '
                                                                             'parameters are 3 1.5 1 0.5')
    parser.add_argument('--kernel', '-k', choices=['bigaussian', 'gaussian'], help='Choose between smoothing kernels, valid '
                                                                                   'options are bigaussian (default) or gaussian.')
    parser.add_argument('--vesselness', '-v', choices=['bigaussian', 'frangi', 'sato'], help='Choose between vesselness functions, valid'
                                                                                             'options are bigaussian (default), frangi or sato.')
    parser.add_argument('--directory', '-d', choices=['y', 'n'], help='If set to \'y\', filter will process every .tif image in the directory')
    parser.add_argument('--zratio', '-z', type=float, help='For anisotropic images, set the scale of the z-axis, typically <1.')
    args = parser.parse_args()
    if args.params is None:
        args.params = [3, 1.5, 1, 0.5]
    if args.kernel == 'gaussian':
        kernel_param = bigaussian.gaussian_kernel_3d
    else:
        kernel_param = bigaussian.bigaussian_kernel_3d_alt
    if args.vesselness == 'sato':
        vesselness_param = bigaussian.lineness_sato_3d
    elif args.vesselness == 'frangi':
        vesselness_param = bigaussian.lineness_frangi_3d
    else:
        vesselness_param = bigaussian.lineness_bg_3d
    if args.zratio is None:
        args.zratio = 1
    if args.directory == 'y':
        process_16bit_folder(args.input, kernel_param, vesselness_param, args.params[0], args.params[1], args.params[3], int(args.params[2]), args.zratio)
    else:
        process_16bit_file(args.input, args.output, kernel_param, vesselness_param, args.params[0], args.params[1], args.params[3], int(args.params[2]), args.zratio)
