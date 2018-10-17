import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.nn.rnn_cell import LSTMCell

# from .cell import LSTMCell, MultiLSTMCell


class DeepLSTM:
    def __init__(self, config,datainfo):
        self.config = config
        self.datainfo = datainfo
        self.add_placeholders()
        self.pred, self.loss, self.train_op = self.build_graph()

    def add_placeholders(self):
        #assume every question and answer have the same length
        #handle one question and all the answer at the same time
        self.question_placeholder = tf.placeholder(tf.int32,[self.config.maxlen, 1])
        #ATTENTION: best answer in the first place
        self.answer_placeholder = tf.placeholder(tf.int32,[self.config.maxlen, None])
        self.score_placeholder = tf.placeholder(tf.int32, [None])
        self.sequence_length = tf.placeholder(tf.int32, [None])
        self.state_keep_prob = tf.placeholder(tf.float32, shape=())

    def create_feed_dict(self, answer,question,sequence_length, state_keep_prob=1.0, score=None):
        feed_dict = {
            self.answer_placeholder: answer,
            self.question_placeholder:question ,
            self.sequence_length: sequence_length
        }
        if labels_batch is not None:
            feed_dict[self.score_placeholder] = score
        return feed_dict

    def generate_cell(self):
        # use ordinary cell
        return rnn.DropoutWrapper(LSTMCell(self.config.numhidden), output_keep_prob=self.state_keep_prob)

    #        return rnn.DropoutWrapper(LSTMCell(self.config.state_size), state_keep_prob=self.state_keep_prob)
    # def user_defined_cell(self):
    #     return rnn.DropoutWrapper(LSTMCell(self.config.state_size), state_keep_prob=self.state_keep_prob)

    def build_graph(self):
        #TODO: set to pre-trained word2vect
        
        embed = tf.get_variable(
            "word_embedding", shape=[self.config.vocab_size, self.config.embed_size], initializer=xavier_initializer()
        )


        question = tf.nn.embedding_lookup(embed, self.question_placeholder)
        answer = tf.nn.embedding_lookup(embed,self.answer_placeholder)
        #depth should be one
        cells = [self.generate_cell() for _ in range(self.config.depth)]
        stack_cells = rnn.MultiRNNCell(cells)
        with tf.variable_scope('train'):
            _, state_quesiton = tf.nn.dynamic_rnn(stack_cells, question, sequence_length=self.sequence_length, dtype=tf.float32)

        #PUZZLE: what will happen if there batch of answer 
        with tf.variable_scope('train',reuse=True):
            _, state_answer = tf.nn.dynamic_rnn(stack_cells, answer, sequence_length=self.sequence_length, dtype=tf.float32)
        
        answer_vector = tf.concat([it[0] for it in state_answer], axis=1, name="h")
        question_vector = tf.concat([it[0] for it in state_question], axis=1, name="h")
        
        #init User Variable
        #TODO: use user profile to initialize user vector
        user = tf.get_variable("user",[self.datainfo.userSize, self.config.userDim])
        Loss = []
        goodAns = df.gather(answer_vector,0)
        goodUser = df.gather(answer_vector, 0)
        # x: count of All answer
        for i in range(1, x):
            a1 = tf.gather(answer_vector,0)
            
        w_0 = tf.get_variable("w_0", [self.config.vocab_size, self.config.state_size * self.config.depth],
                              initializer=xavier_initializer())

        b_0 = tf.get_variable("b_0", [self.config.vocab_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        pred = tf.matmul(h, w_0, transpose_b=True) + b_0

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred,
            labels=self.labels_placeholder,
            name="loss"
        )
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return pred, loss, train_op

    def predict_on_batch(self, sess, inputs_batch, sequence_length):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, sequence_length=sequence_length)
        predictions = sess.run(self.pred, feed_dict=feed)
        return np.argmax(predictions, axis=1)

    def train_on_batch(self, sess, inputs_batch, sequence_length, labels_batch, keep_prob=0.2):
        feed = self.create_feed_dict(
            inputs_batch=inputs_batch, sequence_length=sequence_length,
            state_keep_prob=keep_prob, labels_batch=labels_batch
        )
        _, loss, prediction = sess.run([self.train_op, self.loss, self.pred], feed_dict=feed)
        return loss, np.argmax(prediction, axis=1)
