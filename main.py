# -*- coding: utf-8 -*-
__author__ = 'Vojtech Vozab'
import bigaussian
import argparse
import glob
import numpy as np
import os
from skimage import io


def process_16bit_folder(path, kernel_function=bigaussian.bigaussian_kernel_3d, vesselness_function=bigaussian.lineness_bg_3d, sigma_foreground=3, sigma_background=1.5, step_size=0.5, number_steps=1):
    suffix = 'tif'
    image_list = glob.glob(path + "*" + suffix)
    if not image_list:
        print "no loadable files in folder"
        return -1

    for image_file in image_list:
        process_16bit_file(image_file, None, kernel_function, vesselness_function, sigma_foreground, sigma_background, step_size, number_steps)


def process_16bit_file(input_file, output_file=None, kernel_function=bigaussian.bigaussian_kernel_3d, vesselness_function=bigaussian.lineness_bg_3d, sigma_foreground=3, sigma_background=1.5, step_size=0.5, number_steps=1):
    img_3d_float = io.imread(input_file).astype(np.float64) / 65535
    if output_file is None:
        directory, filename = os.path.split(input_file)
        filename_nosuf, suffix = os.path.splitext(filename)
        if not os.path.exists(os.path.join(directory, "out")):
            os.makedirs(os.path.join(directory, "out"))
        print directory, filename_nosuf, suffix
        output_file = os.path.join(directory, "out", filename_nosuf)+"_out"+suffix

    output_3d_float = bigaussian.general_filter_3d(img_3d_float, kernel_function, vesselness_function, sigma_foreground, sigma_background, step_size, number_steps)
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
    args = parser.parse_args()
    if args.params is None:
        args.params = [3, 1.5, 1, 0.5]
    if args.kernel == 'gaussian':
        kernel_param = bigaussian.gaussian_kernel_3d
    else:
        kernel_param = bigaussian.bigaussian_kernel_3d
    if args.vesselness == 'sato':
        vesselness_param = bigaussian.lineness_sato_3d
    elif args.vesselness == 'frangi':
        vesselness_param = bigaussian.lineness_frangi_3d
    else:
        vesselness_param = bigaussian.lineness_bg_3d
    if args.directory == 'y':
        process_16bit_folder(args.input, kernel_param, vesselness_param, args.params[0], args.params[1], args.params[3], int(args.params[2]))
    else:
        process_16bit_file(args.input, args.output, kernel_param, vesselness_param, args.params[0], args.params[1], args.params[3], int(args.params[2]))
