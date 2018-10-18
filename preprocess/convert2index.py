from tensorflow.contrib.keras import preprocessing 
from gensim.models import KeyedVectors



class Config:
    vocab_size = 100002
    embed_size = 200
    state_size = 256
    depth = 1
    lr = 5E-4
    batch_size = 32
    num_sampled = 512
    n_epoch = 512
    train_steps = 200


class Word2index:
    def __init__(self, text_corpus):
        self.tokenizer = preprocessing.text.Tokenizer(num_words=Config.vocab_size)
        self.tokenizer.fit_on_texts(text_corpus)
    
    def convert(self, text_convert):
        return self.tokenizer.texts_to_sequences(text_convert)
    
    def loadEmbeddingMatrix(self, file_dir):
        word2vec = KeyedVectors.load_word2vec_format(file_dir, binary=True)
        print('Found %s word vectors of word2vec' % len(word2vec.vocab))
        word_index = self.tokenizer.word_index
        nb_words = min(Config.vocab_size, len(word_index))+1
        embedding_matrix = np.zeros((nb_words, Config.embed_size))
        for word, i in word_index.items():
            if word in word2vec.vocab:
                embedding_matrix[i] = word2vec.word_vec(word)
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        return embedding_matrix
    
    





        
