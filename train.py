from Preprocessing.Data_Preparation import *
from Preprocessing.Data_Cleaning import *
from model.Model_Architecture import model_architecture
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

BATCH_SIZE = 64
EPOCHS = 100

if __name__ == "__main__":
    data, labels = read_imdb_data()
    data, eng_word_dict, num_english_token = word2vec(data)

    ######################################################################################
    cache_data = dict(eng_word_dict=eng_word_dict, num_english_token=num_english_token)
    with open('data.pkl', "wb") as f:
        pickle.dump(cache_data, f)
    ######################################################################################

    Y = pd.get_dummies(labels)
    X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.4)

    model = model_architecture(num_english_token)
    parallel_model = multi_gpu_model(model, gpus=2)
    opt = Adam(lr=0.0005)
    parallel_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.001)

    parallel_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1,
                       callbacks=[reduce_lr])
    parallel_model.save('sentiment_2.h5')

    scores = parallel_model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", scores[1])

