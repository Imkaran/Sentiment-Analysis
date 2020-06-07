from Preprocessing.Data_Preparation import *
from Preprocessing.Data_Cleaning import *
from model.Model_Architecture import model_architecture
import pandas as pd
import pickle
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

BATCH_SIZE = 64
EPOCHS = 10

if __name__=="__main__":

    data, labels = read_imdb_data()
    data, eng_word_dict, num_english_token = word2vec(data)

    ######################################################################################
    cache_data = dict(eng_word_dict=eng_word_dict, num_english_token=num_english_token)
    with open('data.pkl', "wb") as f:
        pickle.dump(cache_data, f)
    ######################################################################################

    Y = pd.get_dummies(labels)

    model = model_architecture(num_english_token)
    model.compile(loss="categorical_crossentropy", optimizer= "adam", metrics=['accuracy'])
    model.fit(data, Y, batch_size=BATCH_SIZE, epochs=EPOCHS)
    model.save('sentiment.h5')
    # data_train = word2vec(data_train)

    
