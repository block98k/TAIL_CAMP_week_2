from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from Image_Generator import TextImageGenerator
from Model import get_Model
from parameter import *
K.set_learning_phase(0)

# # Model description and training

model = get_Model(training=True)

try:
    model.load_weights('LSTM+BN5--10--33.831.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

train_file_path = '../talweek2/newtrainkk/'
tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size, downsample_factor)
tiger_train.build_data()

valid_file_path = '../talweek2/newvalidkk/'
tiger_val = TextImageGenerator(valid_file_path, img_w, img_h, val_batch_size, downsample_factor)
tiger_val.build_data()

adam = Adam()


early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=3)
checkpoint = ModelCheckpoint(filepath='LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=3, mode='min', period=5)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

# captures output of softmax so we can decode the output during visualization
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(tiger_train.n / batch_size),
                    epochs=200,
                    callbacks=[checkpoint,TensorBoard(log_dir='log')],
                    validation_data=tiger_val.next_batch(),
                    validation_steps=int(tiger_val.n / val_batch_size))
