from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.optimizers import Adam

import numpy as np


def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img/std
    return img


def extract_roi(img):
    rows, cols = img.shape
    mean = np.mean(img)
    max = np.max(img)
    min = np.min(img)

    img[img == max] = mean
    img[img == min] = mean

    kmeans = KMeans(n_clusters=2).fit(np.reshape(img, [np.prod(img.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())

    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([10, 10]))

    labels = measure.label(dilation)
    label_vals = np.unique(labels)

    regions = measure.regionprops(labels)
    good_labels = []

    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)

    mask = np.ndarray([rows, cols], dtype=np.int8)
    mask[:] = 0

    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)

    if (mask == 1).sum() < 26214:
        return None

    mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation

    img_masked = img * mask
    new_mean = np.mean(img_masked[mask > 0])
    new_std = np.std(img_masked[mask > 0])
    old_min = np.min(img_masked)
    img_masked[img_masked == old_min] = new_mean - 1.2 * new_std  # resetting backgound color
    img_masked = img_masked - new_mean
    img_masked = img_masked / new_std

    labels = measure.label(mask)
    regions = measure.regionprops(labels)

    min_row = 512
    max_row = 0
    min_col = 512
    max_col = 0
    for prop in regions:
        B = prop.bbox
        if min_row > B[0]:
            min_row = B[0]
        if min_col > B[1]:
            min_col = B[1]
        if max_row < B[2]:
            max_row = B[2]
        if max_col < B[3]:
            max_col = B[3]
    width = max_col - min_col
    height = max_row - min_row
    if width > height:
        max_row = min_row + width
    else:
        max_col = min_col + height

    imgc = img_masked[min_row:max_row, min_col:max_col]
    maskc = mask[min_row:max_row, min_col:max_col]

    if max_row - min_row < 5 or max_col - min_col < 5:  # skipping all images with no god regions
        return None
    else:
        mean = np.mean(imgc)
        imgc = imgc - mean
        min = np.min(imgc)
        max = np.max(imgc)
        imgc = imgc / (max - min)
        new_img = resize(imgc, [512, 512], mode='constant')

        return new_img


def filter_good(flag_img):
    flag, img = flag_img
    return flag

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet(img_rows, img_cols):
    K.set_image_dim_ordering('th')
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def get_nodule_mask_extractor(weights_file):
    model = get_unet(512, 512)
    model.load_weights(weights_file)

    def extract_nodule_masks(image):
        rows, cols = image.shape
        mask = model.predict(image.reshape(1, 1, rows, cols))
        return image, mask

    return extract_nodule_masks


def extract_region_from_mask(path, patient, image, mask):
    print(mask.shape)
    labels = measure.label(mask)
    regions = measure.regionprops(labels)

    if len(regions) == 0:
        print("No regions found")
        return path, patient, image, False

    img_masked = image * mask
    new_mean = np.mean(img_masked[mask > 0])
    new_std = np.std(img_masked[mask > 0])
    old_min = np.min(img_masked)
    img_masked[img_masked == old_min] = new_mean - 1.2 * new_std  # resetting backgound color
    img_masked = img_masked - new_mean
    img_masked = img_masked / new_std

    min_row = 512
    max_row = 0
    min_col = 512
    max_col = 0

    for prop in regions:
        B = prop.bbox
        if min_row > B[0]:
            min_row = B[0]
        if min_col > B[1]:
            min_col = B[1]
        if max_row < B[2]:
            max_row = B[2]
        if max_col < B[3]:
            max_col = B[3]
    width = max_col - min_col
    height = max_row - min_row

    if width > height:
        max_row = min_row + width
    else:
        max_col = min_col + height

    img = image[min_row:max_row, min_col:max_col]
    mask = mask[min_row:max_row, min_col:max_col]
    mean = np.mean(img)
    img = img - mean
    min = np.min(img)
    max = np.max(img)
    img = img / (max - min)
    new_img = resize(img, [50, 50], mode='constant')

    return patient, path, new_img, True
