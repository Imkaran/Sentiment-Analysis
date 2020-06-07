from keras.models import load_model
from keras import preprocessing
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from lazyme.string import color_print

def get_cache_data(cache_path = 'data.pkl'):
    with open(cache_path, 'rb') as cache:
        return pickle.load(cache)

if __name__ == '__main__':

    #my_review = "this is very disgusting movie i have ever seen do not watch this movie it is not worth"
    # my_review = "It is very awesome movie i suggest to watch it it give a pleasure"
    my_review = input("Enter your review:")
    model = load_model('sentiment_2.h5')
    cache_data = get_cache_data(cache_path = 'data.pkl')
    eng_word_dict, num_english_token = cache_data['eng_word_dict'], cache_data['num_english_token']

    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(my_review)
    input_corpus = []

    for word in my_review.lower().split(" "):
        if word in eng_word_dict.keys():
            input_corpus.append(eng_word_dict[word])

    input_review = pad_sequences([input_corpus], maxlen=500)

    predictions = model.predict(input_review)

    if np.argmax(predictions) == 0:
        color_print("It's negative review", color='red')
    elif np.argmax(predictions) == 1:
        color_print("It's positive review", color='green')
