# -*- coding: utf-8 -*-
__author__ = 'Vojtech Vozab'
import bigaussian
import argparse
import multiprocessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A lineness filter for 3D images.')
    parser.add_argument('input', help='input filename')
    parser.add_argument('output', help='output filename')
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
    multiprocessing.freeze_support()
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
    bigaussian.general_filter_3d(args.input, args.output, kernel_param, vesselness_param, args.params[0], args.params[1], args.params[3], int(args.params[2]))
