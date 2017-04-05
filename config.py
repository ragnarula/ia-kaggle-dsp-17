import ia_kdsb17.image_prep.helpers as imhelpers
from itertools import starmap, tee
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras.optimizers import SGD


def cropped_averaged_scaled(image_dir, labels_df, **kwargs):
    patients = imhelpers.image_dirs(image_dir)

    patients, p2 = tee(patients)
    train_size = 0
    test_size = 0

    for p in p2:
        if p[1] in labels_df.index:
            train_size += 1
        else:
            test_size += 1

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

    return train_size, train, test_size, test


def get_model(input_shape, **kwargs):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, data_format='channels_last'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


model_params = {
    'batch_size': 128,
    'epochs': 100,
    'shuffle': True
}

model_function = get_model

image_params = {
    'scale': 0.25
}

image_prep_function = cropped_averaged_scaled

