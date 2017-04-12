__author__ = 'vozka'
from bigaussian import *


def convert_to_mha(input, output):
    array2d = sitk.GetArrayFromImage(sitk.ReadImage(input))
    array2d = np.clip(array2d, 0, 255).astype(np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(array2d), output)


def all_filters(input, output, sigma, sigmab, stepsize, nsteps):
    general_filter_3d(input, "./output/"+output+"-sato-bg.mha", bigaussian_kernel_3d, lineness_sato_3d, sigma, sigmab, stepsize, nsteps)
    print "sato-bg done"
    general_filter_3d(input, "./output/"+output+"-frangi-bg.mha", bigaussian_kernel_3d, lineness_frangi_3d, sigma, sigmab, stepsize, nsteps)
    print "frangi-bg done"
    general_filter_3d(input, "./output/"+output+"-bg-bg.mha", bigaussian_kernel_3d, lineness_bg_3d, sigma, sigmab, stepsize, nsteps)
    print "bg-bg done"
    general_filter_3d(input, "./output/"+output+"-sato-gaussian.mha", gaussian_kernel_3d, lineness_sato_3d, sigma, sigmab, stepsize, nsteps)
    print "sato-gaussian done"
    general_filter_3d(input, "./output/"+output+"-frangi-gaussian.mha", gaussian_kernel_3d, lineness_frangi_3d, sigma, sigmab, stepsize, nsteps)
    print "frangi-gaussian done"
    general_filter_3d(input, "./output/"+output+"-bg-gaussian.mha", gaussian_kernel_3d, lineness_bg_3d, sigma, sigmab, stepsize, nsteps)
    print "bg-gaussian done"


def compare_with_ref(reference, output):
    print "sato - bigaussian:"
    print "sensitivity, specificity:", tprtnr(reference, output+"-sato-bg_out_threshold.mha")
    print "hausdorff distance =", hausdorff_distance(reference, output+"-sato-bg_out_threshold.mha")
    print "modified hausdorff distance =", modified_hausdorff_distance(reference, output+"-sato-bg_out_threshold.mha")
    print ""
    print "sato - gaussian:"
    print "sensitivity, specificity:", tprtnr(reference, output+"-sato-gaussian_out_threshold.mha")
    print "hausdorff distance =", hausdorff_distance(reference, output+"-sato-gaussian_out_threshold.mha")
    print "modified hausdorff distance =", modified_hausdorff_distance(reference, output+"-sato-gaussian_out_threshold.mha")
    print ""
    print "frangi - bigaussian:"
    print "sensitivity, specificity:", tprtnr(reference, output+"-frangi-bg_out_threshold.mha")
    print "hausdorff distance =", hausdorff_distance(reference, output+"-frangi-bg_out_threshold.mha")
    print "modified hausdorff distance =", modified_hausdorff_distance(reference, output+"-frangi-bg_out_threshold.mha")
    print ""
    print "frangi - gaussian:"
    print "sensitivity, specificity:", tprtnr(reference, output+"-frangi-gaussian_out_threshold.mha")
    print "hausdorff distance =", hausdorff_distance(reference, output+"-frangi-gaussian_out_threshold.mha")
    print "modified hausdorff distance =", modified_hausdorff_distance(reference, output+"-frangi-gaussian_out_threshold.mha")
    print ""
    print "bg - bigaussian:"
    print "sensitivity, specificity:", tprtnr(reference, output+"-bg-bg_out_threshold.mha")
    print "hausdorff distance =", hausdorff_distance(reference, output+"-bg-bg_out_threshold.mha")
    print "modified hausdorff distance =", modified_hausdorff_distance(reference, output+"-bg-bg_out_threshold.mha")
    print ""
    print "bg - gaussian:"
    print "sensitivity, specificity:", tprtnr(reference, output+"-bg-gaussian_out_threshold.mha")
    print "hausdorff distance =", hausdorff_distance(reference, output+"-bg-gaussian_out_threshold.mha")
    print "modified hausdorff distance =", modified_hausdorff_distance(reference, output+"-bg-gaussian_out_threshold.mha")


def hausdorff_distance(source, target):
    """Computes the Hausdorff distance of two monochromatic images."""
    source_array = sitk.GetArrayFromImage(sitk.ReadImage(source))
    target_array = sitk.GetArrayFromImage(sitk.ReadImage(target))
    source_list = np.argwhere(source_array)
    target_list = np.argwhere(target_array)
    end_dist_ab = np.max(np.amin(spatial.distance.cdist(source_list, target_list), axis=0))
    end_dist_ba = np.max(np.amin(spatial.distance.cdist(target_list, source_list), axis=0))
    return max(end_dist_ab, end_dist_ba)


def modified_hausdorff_distance(source, target):
    """Computes the modified Hausdorff distance of two monochromatic images."""
    source_array = sitk.GetArrayFromImage(sitk.ReadImage(source))
    target_array = sitk.GetArrayFromImage(sitk.ReadImage(target))
    source_list = np.argwhere(source_array)
    target_list = np.argwhere(target_array)
    end_dist_ab = np.median(np.amin(spatial.distance.cdist(source_list, target_list), axis=0))
    end_dist_ba = np.median(np.amin(spatial.distance.cdist(target_list, source_list), axis=0))
    return max(end_dist_ab, end_dist_ba)


def tprtnr(source, filtered):
    """Prints and returns sensitivity and specificity. Source is a perfect (thresholded) output, filtered is a
    thresholded image being compared."""
    source_array = sitk.GetArrayFromImage(sitk.ReadImage(source))/255
    filtered_array = sitk.GetArrayFromImage(sitk.ReadImage(filtered))/255
    tp = np.sum(np.logical_and(source_array == 1, filtered_array == 1)).astype(np.float)
    tn = np.sum(np.logical_and(source_array == 0, filtered_array == 0)).astype(np.float)
    fp = np.sum(np.logical_and(source_array == 0, filtered_array == 1)).astype(np.float)
    fn = np.sum(np.logical_and(source_array == 1, filtered_array == 0)).astype(np.float)
    tpr = (tp/(tp+fn))*100
    tnr = (tn/(tn+fp))*100

    return tpr, tnr

# convert_to_mha("./input/op4.tif", "./input/real_cropped/op4.mha")
# reference_img = "./input/reference.mha"
# input_img = "./input/real_cropped/phosphodefective_cell.tif"

# all_filters(input_img, "real/p_d_c_2/phosphodefective_cell", 1, 0.5, 0.5, 3)
#compare_with_ref(reference_img, "./output/generated_original/lgauss5_single/lgauss5")
