import ia_kdsb17.image_prep.helpers as imhelpers
from itertools import starmap, tee
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras.optimizers import SGD
from sklearn.metrics import log_loss
import ia_kdsb17.image_prep.unet as unet
import os
from keras import backend as K
from skimage.transform import resize


# def cropped_averaged_scaled(image_dir, labels_df, **kwargs):
#     patients = imhelpers.image_dirs(image_dir)
#
#     images = starmap(imhelpers.patient_image, patients)
#     images = starmap(lambda path, patient, im_list: (path, patient, filter(lambda x: len(x) > 0, im_list)), images)
#     dicoms = starmap(imhelpers.patient_dicoms, images)
#     dicoms = starmap(imhelpers.patient_z_pixels, dicoms)
#     dicoms = starmap(imhelpers.patient_z_cropped_pixels, dicoms)
#     dicoms = starmap(imhelpers.patient_z_pixels_sorted, dicoms)
#
#     dicoms = starmap(imhelpers.apply_to_images(imhelpers.drop_first), dicoms)
#     dicoms = starmap(imhelpers.apply_to_images(imhelpers.zero_border), dicoms)
#     dicoms = starmap(imhelpers.patient_averaged_pixels, dicoms)
#     dicoms = starmap(lambda p, pp, i: (p, pp, imhelpers.get_scaler(kwargs['scale'], 2)(i)), dicoms)
#
#     dicoms = starmap(imhelpers.drop_path, dicoms)
#     train, test = tee(dicoms)
#     train = filter(lambda x: x[0] in labels_df.index, train)
#     test = filter(lambda x: x[0] not in labels_df.index, test)
#
#     return train, test


def unet_roi(image_dir, labels_df, **kwargs):

    stream = imhelpers.image_dirs(image_dir)
    stream = starmap(imhelpers.list_images_for_patient, stream)
    stream = starmap(lambda path, patient, im_list: (path, patient, filter(lambda x: len(x) > 0, im_list)), stream)
    stream = starmap(imhelpers.load_dicoms, stream)
    stream = starmap(lambda p, idd, d: (p, idd, imhelpers.dicom_to_z_pixel(d)), stream)
    stream = starmap(lambda p, idd, d: (p, idd, imhelpers.sort_images(d)), stream)
    stream = imhelpers.flat_starmap(lambda p, idd, z_ims: starmap(lambda z, im: (p, idd, im), z_ims), stream)
    stream = starmap(lambda p, idd, im: (p, idd, unet.standardize(im)), stream)
    stream = starmap(lambda p, idd, im: (p, idd, unet.extract_roi(im)), stream)
    stream = filter(lambda x: x[2] is not None, stream)
    stream = starmap(lambda p, idd, im: (p, idd, resize(im, [256, 256], mode='constant')), stream)
    stream = starmap(lambda p, idd, im: (p, idd, im.reshape(1, im.shape[0], im.shape[1])), stream)
    stream = starmap(lambda p, idd, im: ("train", idd, im) if idd in labels_df.index else ("test", idd, im), stream)

    return stream


def unet_nodules(image_dir, labels_df, **kwargs):
    stream = imhelpers.image_dirs(image_dir)
    stream = starmap(imhelpers.list_images_for_patient, stream)
    stream = starmap(lambda path, patient, im_list: (path, patient, filter(lambda x: len(x) > 0, im_list)), stream)
    stream = starmap(imhelpers.load_dicoms, stream)
    stream = starmap(lambda p, idd, d: (p, idd, imhelpers.dicom_to_z_pixel(d)), stream)
    stream = starmap(lambda p, idd, d: (p, idd, imhelpers.sort_images(d)), stream)
    stream = imhelpers.flat_starmap(lambda p, idd, z_ims: starmap(lambda z, im: (p, idd, im), z_ims), stream)
    stream = starmap(lambda p, idd, im: (p, idd, unet.standardize(im)), stream)
    stream = starmap(lambda p, idd, im: (p, idd, unet.extract_roi(im)), stream)
    stream = filter(lambda x: x[2] is not None, stream)
    stream = starmap(lambda p, idd, im: (p, idd, unet.get_nodule_mask_extractor(kwargs['weights_file'])(im)), stream)
    stream = filter(lambda x: x[2][1] is not None, stream)
    stream = starmap(lambda p, idd, im_msk: (p, idd, im_msk[0] * im_msk[1].reshape(512, 512)), stream)
    stream = starmap(lambda p, idd, im: (p, idd, resize(im, [256, 256], mode='constant')), stream)

    stream = starmap(lambda p, idd, im: (p, idd, im.reshape(1, im.shape[0], im.shape[1])), stream)

    stream = starmap(lambda p, idd, im: ("train", idd, im) if idd in labels_df.index else ("test", idd, im), stream)

    return stream


def get_model(input_shape, **kwargs):
    K.set_image_dim_ordering('th')
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (1, 1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


model_params = {
    'batch_size': 128,
    'epochs': 1,
    'shuffle': True
}

model_function = get_model

weights_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unet.hdf5')

image_params = {
    'scale': 0.25,
    'weights_file': weights_file
}

image_prep_function = unet_nodules
validation_function = log_loss
