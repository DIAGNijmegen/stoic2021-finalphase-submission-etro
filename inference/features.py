
import SimpleITK as sitk


def lesion_fractions(segmentation):
    label_filter = sitk.LabelShapeStatisticsImageFilter()
    label_filter.Execute(segmentation)
    labels = label_filter.GetLabels()

    # Can only calculate the volume if the label is present
    if 1 in labels:
        volume_ggo = label_filter.GetPhysicalSize(1) / 1000  # GetPhysicalSize is in mm3 so /1000 to get ml
    else:
        volume_ggo = 0
    if 2 in labels:
        volume_cons = label_filter.GetPhysicalSize(2) / 1000
    else:
        volume_cons = 0
    if 3 in labels:
        volume_lung = label_filter.GetPhysicalSize(3) / 1000
    else:
        volume_lung = 0
    volume_total = volume_ggo + volume_cons + volume_lung

    fraction_ggo = volume_ggo / volume_total
    fraction_cons = volume_cons / volume_total

    return fraction_ggo, fraction_cons


def intensity_and_texture(preprocessed_image, segmentation):
    labelstatsFilter = sitk.LabelIntensityStatisticsImageFilter()
    segmentation.CopyInformation(preprocessed_image)
    labelstatsFilter.Execute(segmentation, preprocessed_image)
    labels = labelstatsFilter.GetLabels()
    if 3 in labels:  # should always be there, but just in case...
        mean_healthy = labelstatsFilter.GetMean(3)
        kurtosis_healthy = labelstatsFilter.GetKurtosis(3)
        skewness_healthy = labelstatsFilter.GetSkewness(3)
    else:
        mean_healthy = -1
        kurtosis_healthy = 6.1      # median value from the training data
        skewness_healthy = 2.3      # median value from the training data
    if 1 in labels:
        mean_ggo = labelstatsFilter.GetMean(1)
        kurtosis_ggo = labelstatsFilter.GetKurtosis(1)
        skewness_ggo = labelstatsFilter.GetSkewness(1)
    else:
        mean_ggo = -1  # healthy lung
        kurtosis_ggo = kurtosis_healthy
        skewness_ggo = skewness_healthy
    if 2 in labels:
        mean_cons = labelstatsFilter.GetMean(2)
        kurtosis_cons = labelstatsFilter.GetKurtosis(2)
        skewness_cons = labelstatsFilter.GetSkewness(2)
    else:
        mean_cons = -1
        kurtosis_cons = kurtosis_healthy
        skewness_cons = skewness_healthy

    return mean_healthy, mean_ggo, mean_cons, \
           kurtosis_healthy, kurtosis_ggo, kurtosis_cons, \
           skewness_healthy, skewness_ggo, skewness_cons
