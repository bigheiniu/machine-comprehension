import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.nn.rnn_cell import LSTMCell

# from .cell import LSTMCell, MultiLSTMCell
#TODO: config add bias, maxlens


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
        self.answer_length_list = tf.placeholder(tf.int32, [None])
        self.answer_user_placeholder = tf.placeholder(tf.int32,[None])
        self.answer_vote_placeholder = tf.placeholder(tf.float32,[None])
        self.state_keep_prob = tf.placeholder(tf.float32, shape=())
        self.question_length = tf.placeholder(tf.int32, shape=())

    def create_feed_dict(self, answer,question,answer_length_list,answer_user_list, answer_vote_list, state_keep_prob=1.0, question_length,score=None):
        feed_dict = {
            self.answer_placeholder: answer,
            self.question_placeholder:question ,
            self.answer_length_list: answer_length_list
            self.question_length: question_length
            self.answer_user_placeholder: answer_user_list
            self.answer_vote_placeholder: answer_vote_list
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
            "word_embedding", shape=[self.config.vocab_size, self.config.embed_size], initializer=xavier_initializer(),trainable=False
        )


        question = tf.nn.embedding_lookup(embed, self.question_placeholder)
        answer = tf.nn.embedding_lookup(embed,self.answer_placeholder)
        #depth should be one
        cells = [self.generate_cell() for _ in range(self.config.depth)]
        stack_cells = rnn.MultiRNNCell(cells)
        with tf.variable_scope('train'):
            question_lstm_output,_ = tf.nn.dynamic_rnn(stack_cells, question, sequence_length=self.question_length, dtype=tf.float32)

        #PUZZLE: what will happen if there batch of answer 
        with tf.variable_scope('train',reuse=True):
            answer_lstm_ouput,_  = tf.nn.dynamic_rnn(stack_cells, answer, sequence_length=self.answer_length_list, dtype=tf.float32)
        
        # transpose [1, 0, 2] means the original one is [0, 1, 2] and we swap the first 2 dimensions.
        # swap the batch_size with sequence size.
        question_lstm_output = tf.transpose(question_lstm_output, [1, 0, 2])
        answer_lstm_ouput = tf.transpose(answer_lstm_ouput, [1, 0, 2])
        question_lstm_output = tf.reduce_mean(question_lstm_output,axis=0)
        answer_lstm_ouput = tf.reduce_mean(answer_lstm_ouput,axis=0)  
        
        #init User Variable
        #TODO: use user profile to initialize user vector
        user = tf.get_variable("user",[self.datainfo.userSize, self.config.userDim], initializer=xavier_initializer())
        Loss = []
        a_good = df.gather(answer_lstm_ouput,tf.convert_to_tensor(0))
        u_good = df.gather(self.answer_user_placeholder, tf.convert_to_tensor(0))
        c = tf.constant(self.config.constant_bias)
        a_size = tf.shape(answer)[1]
        # x: count of All answer
        for i_tensor in tf.range(1, a_size):
            a_bad = tf.gather(answer_vector,i_tensor)
            u_bad = tf.gather(user, tf.gather(answer_user_placeholder, i_tensor))
            Loss.append( c + tf.matmul(question_lstm_output, a_bad, transpose_b=True) \ 
                * tf.matmul(question_lstm_output, u_bad, transpose_b=True) \
                - tf.matmul(question_lstm_output, a_good,transpose_b=True) * tf.matmul(question_lstm_output, u_good, transpose_b=True) )
        Loss = tf.stack(Loss)
        compareValue = tf.zeros([1,1])
        mask = tf.less(Loss, compareValue)
        
        pred =  (tf.size(tf.boolean_mask(Loss,mask)) > 0) and tf.convert_to_tensor(0) or tf.convert_to_tensor(1)
        Loss = tf.map_fn(lambda x: tf.maxmize(compareValue, x), Loss)

        #relation norm
        all_vote = tf.reduce_sum(self.answer_vote_placeholder)
        vote_weight = tf.divide(self.answer_vote_placeholder, all_vote)
        norm = []
        index = 0
        for user_id in self.answer_user_placeholder:
            index_tensor = tf.convert_to_tensor(index)
            u1_vote_norm = tf.gather(self.answer_vote_placeholder, index_tensor)
            u1_vec = tf.gather(user, user_id)
            u2_matrix = tf.gather(user, answer_user_placeholder)
            norm.append( u1_vec - tf.reduce_sum(tf.multiply(u1_vote_norm , u2_matrix )))
            index = index + 1
        norm = tf.stack(norm)
        norm = tf.reduce_sum(norm)
        error_norm = tf.reduce_sum(Loss) + self.config.Lambda * export_norm
        #accuracy evaluation

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred,
            labels=self.labels_placeholder,
            name="loss"
        )
        optimizer = tf.train.AdamOptimizer()
        minimize = optimizer.minimize(error_norm)
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(error_norm)
        return pred, loss, train_op

    def predict_on_batch(self, sess, inputs_batch, answer_length_list):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, answer_length_list=answer_length_list)
        predictions = sess.run(self.pred, feed_dict=feed)
        return np.argmax(predictions, axis=1)

    def train_on_batch(self, sess, inputs_batch, answer_length_list, labels_batch, keep_prob=0.2):
        feed = self.create_feed_dict(
            inputs_batch=inputs_batch, answer_length_list=answer_length_list,
            state_keep_prob=keep_prob, labels_batch=labels_batch
        )
        _, loss, prediction = sess.run([self.train_op, self.loss, self.pred], feed_dict=feed)
        return loss, np.argmax(prediction, axis=1)
