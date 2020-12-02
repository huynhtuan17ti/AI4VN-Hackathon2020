import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout,BatchNormalization
from tensorflow.keras.models  import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
import efficientnet.tfkeras as efn

from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

BATCH_SIZE = 80
IMG_SIZE = 512
NUM_CLASSES = 7
ROOT_PATH = '/home/ryan/Machine_Learning/AI4VN-Final'
MODEL_PATH = '/home/ryan/Machine_Learning/AI4VN-Final/models'
SUBMISSION_PATH = '/home/ryan/Machine_Learning/AI4VN-Final/submissions'
threshold = 0.38

def get_generator(_datagen_test, test_df):
    test_generator = _datagen_test.flow_from_dataframe(
                dataframe=test_df,
                directory=ROOT_PATH +'/'+"test_set",
                x_col="image_id",
                y_col=None,
                has_ext=True,
                class_mode=None,
                batch_size=BATCH_SIZE,
                seed=42,
                shuffle=False,
                target_size=(IMG_SIZE, IMG_SIZE))
    return test_generator

def get_model():
    base_model =  efn.EfficientNetB6(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    _x = (Dropout(0.3))(x)
    predictions = Dense(NUM_CLASSES, activation="sigmoid")(_x)
    model =  Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(os.path.join(MODEL_PATH, "EfnB6_multi_label_model_combine_2.h5"))
    print('Load model sucessfully !')
    return model

def to_submission(test_df, y_pred):
    image_id = []
    label = []
    for i in range(len(test_df)):
        image_id.append(test_df['image_id'][i])
        if max(y_pred[i]) < threshold:
            label.append(0)
        else:
            label.append(np.argmax(y_pred[i])+1)
    dict = {'image_id': image_id, 'label': label}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(SUBMISSION_PATH, "submission.csv"), index = False, header = False, sep = '\t')
    print('Make submission file sucessfully !')

def main():
    test_df = pd.read_csv(ROOT_PATH + '/' + "test.csv")
    _datagen_test = ImageDataGenerator(rescale = 1./255.)
    test_generator = get_generator(_datagen_test, test_df)
    model = get_model()
    y_pred = model.predict_generator(test_generator, verbose = 1, workers = 4, use_multiprocessing = True)
    to_submission(test_df, y_pred)
    

if __name__ == '__main__':
    main()

