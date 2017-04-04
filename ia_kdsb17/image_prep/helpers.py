import os
import dicom
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, starmap


def flatmap(f, items):
    return chain.from_iterable(map(f, items))


# (path, patient_id)
def image_dirs(dir):
    ids = os.listdir(dir)[1:]
    for patient_id in ids:
        if os.path.isdir(os.path.join(dir, patient_id)):
            yield (dir, patient_id)


# (path, patient, [image, image ...])
def patient_image(path, patient):
    full_path = os.path.join(path, patient)
    images = os.listdir(full_path)[1:]
    return path, patient, images


# (path, patient, [dicom, dicom, dicom]
def patient_dicoms(path, patient, images):
    slices = map(lambda s: dicom.read_file(os.path.join(path, patient, s)), images)
    return path, patient, slices


# (path, patient, [(z, pixels)])
def patient_z_pixels(path, patient, dicoms):
    slices = map(lambda s: (int(s.ImagePositionPatient[2]), s.pixel_array), dicoms)
    return path, patient, slices


# (path, patient, [(z, pixels)])
def patient_z_cropped_pixels(path, patient, z_dicoms):
    slices = starmap(lambda z, d: (z, d[90:422, :]), z_dicoms)
    return path, patient, slices


# (path, patient, [(z, pixels)])
def patient_z_pixels_sorted(path, patient, z_dicoms):
    slices = iter(sorted(z_dicoms, key=lambda x: x[0]))
    return path, patient, slices


def zero_border(img):
    img[img == -2000] = 0
    return img


def drop_first(img):
    return img[1]


def mean_normalize(img):
    norm = img - img.mean()
    # norm = norm / norm.max()
    return norm


def apply_to_images(f):
    def applier(path, patient, pixels):
        mapped = map(f, pixels)
        return path, patient, mapped
    return applier


# (path, patient, pixels)
def patient_averaged_pixels(path, patient, pixels):

    list_pixels = list(pixels)
    num_images = len(list_pixels)

    h, w = list_pixels[0].shape
    sum_image = np.zeros((h, w), dtype=np.float64)

    for p in list_pixels:
        p = p / num_images
        sum_image += p

    return path, patient, sum_image


def drop_path(path, patient, pixels):
    return patient, pixels


def patient_average(dir):
    patients = image_dirs(dir)
    images = starmap(patient_image, patients)
    dicoms = starmap(patient_dicoms, images)
    dicoms = starmap(patient_z_pixels, dicoms)
    dicoms = starmap(patient_z_cropped_pixels, dicoms)
    dicoms = starmap(patient_z_pixels_sorted, dicoms)

    dicoms = starmap(apply_to_images(drop_first), dicoms)
    dicoms = starmap(apply_to_images(zero_border), dicoms)
    # dicoms = starmap(apply_to_images(mean_normalize), dicoms)

    dicoms = starmap(patient_averaged_pixels, dicoms)

    dicoms = starmap(drop_path, dicoms)
    return dicoms

# for testing...
if __name__ == "__main__":
    dicoms = patient_average("/Users/rag/code/ragnarula/kaggle-2017/data")
    one_patient = next(dicoms)
    print(one_patient)
    fig = plt.figure()
    # for image in one_patient[2]:
    #     print(image)
    #     plt.imshow(image, cmap=plt.cm.bone)
    #     plt.show()

    plt.imshow(one_patient[1], cmap=plt.cm.bone)
    plt.show()