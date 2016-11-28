# -*- coding: utf-8 -*-
__author__ = 'Vojtech Vozab'
import bigaussian
import argparse
import multiprocessing
import os.path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A lineness filter for 2D and 3D images.')
    parser.add_argument('input', help='filename')
    parser.add_argument('output')
    parser.add_argument('--dimensions', '-d', choices=['2d', '3d', '2D', '3D'], help='Switch between 2D mode and 3D mode. When '
                                                                           'omitted, files ending with .mha are expected'
                                                                           'to be 3D, everything else 2D.')
    parser.add_argument('--params', '-p', metavar='X', type=float, nargs=4, help='Filter parameters - foreground sigma and '
                                                                             'background sigma for bigaussian, number of '
                                                                             'multiscale steps and the sigma size by which'
                                                                             'the bigaussian gets enlarged each step, '
                                                                             'in this order. If omitted, the default '
                                                                             'parameters are 1.0 0.4 3 0.2')
    parser.add_argument('--multiprocessing', '-m', choices=['y', 'n'], default='y', help='Switch multiprocessing off/on. Parallel '
                                                                            'version is in general faster for multiscale computations '
                                                                            'that take more than a couple seconds but needs'
                                                                            ' x times more memory, where x is the number'
                                                                            'of multiscale steps.')
    multiprocessing.freeze_support()
    bigaussian.filter2d("./input/hgauss15.png", "hgauss-np.jpg", sigma_foreground=2, sigma_background=0.8, step_size=1, number_steps=3)
    bigaussian.parallel_filter2d("./input/hgauss15.png", "hgauss-p.jpg", sigma_foreground=2, sigma_background=0.8, step_size=1, number_steps=3)
    bigaussian.filter3d("./input/hgauss15.mha", "hgauss15-np.mha", sigma_foreground=1, sigma_background=0.4, step_size=0.6, number_steps=4)
    bigaussian.parallel_filter3d("./input/hgauss15.mha", "hgauss15-p.mha", sigma_foreground=1, sigma_background=0.4, step_size=0.6, number_steps=4)
    args = parser.parse_args()
    if args.dimensions is None:
        extension = os.path.splitext(args.input)[1].strip().lower()
        if extension == '.mha':
            args.dimensions = '3D'
        else:
            args.dimensions = '2D'
    if args.dimensions.upper() == '2D':
        if args.params is None:
            if args.multiprocessing == 'y':
                bigaussian.parallel_filter2d(args.input, args.output)
            else:
                bigaussian.filter2d(args.input, args.output)
        else:
            if args.multiprocessing == 'y':
                bigaussian.parallel_filter2d(args.input, args.output, args.params[0], args.params[1], args.params[3], int(args.params[2]))
            else:
                bigaussian.filter2d(args.input, args.output, args.params[0], args.params[1], args.params[3], int(args.params[2]))
    if args.dimensions.upper() == '3D':
        if args.params is None:
            if args.multiprocessing == 'y':
                bigaussian.parallel_filter3d(args.input, args.output)
            else:
                bigaussian.filter3d(args.input, args.output)
        else:
            if args.multiprocessing == 'y':
                bigaussian.parallel_filter3d(args.input, args.output, args.params[0], args.params[1], args.params[3], int(args.params[2]))
            else:
                bigaussian.filter3d(args.input, args.output, args.params[0], args.params[1], args.params[3], int(args.params[2]))
