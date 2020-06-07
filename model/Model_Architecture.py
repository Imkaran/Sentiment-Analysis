
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

def model_architecture(vocabulary_size):
    
    max_words = 500

    model = Sequential()
    model.add(Embedding(vocabulary_size, 64, input_length=max_words))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(50,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(20))
    model.add(Dense(2, activation='softmax'))

    print(model.summary())

    return model
