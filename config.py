import ia_kdsb17.image_prep.helpers as imhelpers
from itertools import starmap, tee
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras.optimizers import SGD
from sklearn.metrics import log_loss
import ia_kdsb17.image_prep.unet as unet
import os


def cropped_averaged_scaled(image_dir, labels_df, **kwargs):
    patients = imhelpers.image_dirs(image_dir)

    images = starmap(imhelpers.patient_image, patients)
    images = starmap(lambda path, patient, im_list: (path, patient, filter(lambda x: len(x) > 0, im_list)), images)
    dicoms = starmap(imhelpers.patient_dicoms, images)
    dicoms = starmap(imhelpers.patient_z_pixels, dicoms)
    dicoms = starmap(imhelpers.patient_z_cropped_pixels, dicoms)
    dicoms = starmap(imhelpers.patient_z_pixels_sorted, dicoms)

    dicoms = starmap(imhelpers.apply_to_images(imhelpers.drop_first), dicoms)
    dicoms = starmap(imhelpers.apply_to_images(imhelpers.zero_border), dicoms)
    dicoms = starmap(imhelpers.patient_averaged_pixels, dicoms)
    dicoms = starmap(lambda p, pp, i: (p, pp, imhelpers.get_scaler(kwargs['scale'], 2)(i)), dicoms)

    dicoms = starmap(imhelpers.drop_path, dicoms)
    train, test = tee(dicoms)
    train = filter(lambda x: x[0] in labels_df.index, train)
    test = filter(lambda x: x[0] not in labels_df.index, test)

    return train, test


def nodules_unet(image_dir, labels_df, **kwargs):
    patients = imhelpers.image_dirs(image_dir)
    images = starmap(imhelpers.patient_image, patients)
    images = starmap(lambda path, patient, im_list: (path, patient, filter(lambda x: len(x) > 0, im_list)), images)
    dicoms = starmap(imhelpers.patient_dicoms, images)
    dicoms = starmap(imhelpers.patient_z_pixels, dicoms)
    dicoms = starmap(imhelpers.patient_z_pixels_sorted, dicoms)

    dicoms = starmap(imhelpers.apply_to_images(imhelpers.drop_first), dicoms)
    dicoms = starmap(imhelpers.apply_to_images(unet.standardize), dicoms)
    dicoms = starmap(imhelpers.apply_to_images(unet.extract_roi), dicoms)
    dicoms = starmap(imhelpers.filter_images(unet.filter_good), dicoms)
    dicoms = imhelpers.flat_starmap(imhelpers.patient_pixels_list, dicoms)
    dicoms = starmap(unet.get_nodule_mask_extractor(kwargs['weights_file']), dicoms)

    dicoms = starmap(unet.extract_region_from_mask, dicoms)

    dicoms = filter(lambda x: x[3], dicoms)

    dicoms = starmap(imhelpers.drop_path, dicoms)

    train, test = tee(dicoms)
    train = filter(lambda x: x[0] in labels_df.index, train)
    test = filter(lambda x: x[0] not in labels_df.index, test)

    return train, test


def get_model(input_shape, **kwargs):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, data_format='channels_last'))
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

image_prep_function = nodules_unet
validation_function = log_loss
