import os
import dicom
import numpy as np
import scipy.ndimage
from itertools import chain, starmap


def flat_starmap(f, items):
    return chain.from_iterable(starmap(f, items))


# (path, patient_id)
def image_dirs(dir):
    ids = os.listdir(dir)[1:]
    for patient_id in ids:
        if os.path.isdir(os.path.join(dir, patient_id)):
            yield (dir, patient_id)


# (path, patient, [image, image ...])
def list_images_for_patient(path, patient):
    full_path = os.path.join(path, patient)
    images = os.listdir(full_path)[1:]
    return path, patient, images


# (path, patient, [dicom, dicom, dicom]
def load_dicoms(path, patient, images):
    slices = map(lambda s: dicom.read_file(os.path.join(path, patient, s)), images)
    return path, patient, slices


def dicom_to_z_pixel(dicoms):
    return map(lambda s: (int(s.ImagePositionPatient[2]), s.pixel_array), dicoms)


# (path, patient, [(z, pixels)])
def crop_pixels(dicoms):
    return map(lambda d: d[90:422, :])


# (path, patient, [(z, pixels)])
def sort_images(z_dicoms):
    return iter(sorted(z_dicoms, key=lambda x: x[0]))


def patient_pixel_tuples(path, patient, z_dicoms):
    return map(lambda x: (path, patient, x[1]), z_dicoms)


def zero_border(img):
    img[img == -2000] = 0
    return img


def drop_first(img):
    return img[1]


def mean_normalize(img):
    norm = img - img.mean()
    # norm = norm / norm.max()
    return norm


def get_scaler(x, ord):

    def scale(img):
        return scipy.ndimage.zoom(img, x, order=ord)

    return scale


def filter_images(f):
    def filterer(path, patient, pixels):
        filtered = filter(f, pixels)
        return path, patient, filtered
    return filterer


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


def drop_path(_, patient, pixels):
    return patient, pixels




# for testing...
# if __name__ == "__main__":
#     dicoms = patient_average("/Users/rag/code/ragnarula/kaggle-2017/data")
#     one_patient = next(dicoms)
#     print(one_patient)
#     fig = plt.figure()
#     # for image in one_patient[2]:
#     #     print(image)
#     #     plt.imshow(image, cmap=plt.cm.bone)
#     #     plt.show()
#
#     plt.imshow(one_patient[1], cmap=plt.cm.bone)
#     plt.show()