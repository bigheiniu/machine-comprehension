from tensorflow.contrib.keras import preprocessing
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

def padding(sequence, maxlen, dtype='int32', padding='post', trunc='post', value=0):
    return preprocessing.sequence.pad_sequences(sequence, maxlen, dtype, padding, trunc, value)

def padding_easy(sequence, maxlen):
    print(sequence)
    return preprocessing.sequence.pad_sequences(sequence, maxlen)