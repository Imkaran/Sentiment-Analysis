from keras import preprocessing

def word2vec(english_data):
    ###################################################################################

    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(english_data)
    tokenized_english_lines = tokenizer.texts_to_sequences(english_data)

    # sentence_length_list = [len(sentence) for sentence in tokenized_english_lines]
    # max_length_sentence = max(sentence_length_list)

    max_length_sentence = 500
    encoder_input_data = preprocessing.sequence.pad_sequences(tokenized_english_lines, maxlen=max_length_sentence,
                                                              padding='post')

    eng_word_dict = tokenizer.word_index
    num_english_token = len(eng_word_dict)

    return encoder_input_data, eng_word_dict, num_english_token
