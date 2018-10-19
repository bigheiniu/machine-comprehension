import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer


# from .cell import LSTMCell, MultiLSTMCell
#TODO: config add bias, maxlens


class DeepLSTM:
    def __init__(self, config, datainfo, pre_embed=None):
        self.config = config
        self.datainfo = datainfo
        self.pre_embed = pre_embed
        self.add_placeholders()
        self.pred, self.loss, self.train_op = self.build_graph()

    def add_placeholders(self):
        #assume every question and answer have the same length
        #handle one question and all the answer at the same time
        # batch_size * max_length
        self.question_placeholder = tf.placeholder(tf.int32,[None, None])
        #ATTENTION: best answer in the first place
        self.answer_placeholder = tf.placeholder(tf.int32,[None, None])
        # self.score_placeholder = tf.placeholder(tf.int32, [None])
        self.answer_length_list = tf.placeholder(tf.int32, [None])
        self.answer_user_placeholder = tf.placeholder(tf.int32,[None])
        self.answer_vote_placeholder = tf.placeholder(tf.float32,[None])
        self.state_keep_prob = tf.placeholder(tf.float32, shape=())
        self.question_length = tf.placeholder(tf.int32, [None])

    def create_feed_dict(self, answer,question,answer_length_list,answer_user_list, answer_vote_list, question_length, state_keep_prob=1.0 ):
        feed_dict = {
            self.answer_placeholder: answer,
            self.question_placeholder:question ,
            self.answer_length_list: answer_length_list,
            self.question_length: question_length,
            self.answer_user_placeholder: answer_user_list,
            self.answer_vote_placeholder: answer_vote_list,
            self.state_keep_prob: state_keep_prob
        }
        return feed_dict
        # if labels_batch is not None:
        #     feed_dict[self.score_placeholder] = score
        # return feed_dict

    def generate_cell(self):
        # use ordinary cell
        return rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.config.numhidden), output_keep_prob=self.state_keep_prob)

    #        return rnn.DropoutWrapper(LSTMCell(self.config.state_size), state_keep_prob=self.state_keep_prob)
    # def user_defined_cell(self):
    #     return rnn.DropoutWrapper(LSTMCell(self.config.state_size), state_keep_prob=self.state_keep_prob)

    def build_graph(self):
        #TODO: set to pre-trained word2vect
        #############################################
        ### Embedding word index into pre trained word2vect
        #############################################
        embed = tf.get_variable(
            "word_embedding", shape=[self.config.vocab_size, self.config.embed_size], initializer=xavier_initializer()
        )
        question = tf.nn.embedding_lookup(embed, self.question_placeholder)
        answer = tf.nn.embedding_lookup(embed, self.answer_placeholder)
        
        #############################################
        ### Build LSTM structure
        #############################################
        cells = [self.generate_cell() for _ in range(self.config.depth)]
        stack_cells = rnn.MultiRNNCell(cells)
        


        ############################r##############
        ### Question and answer word vector -> LSTM
        ##########################################
        with tf.variable_scope('train'):
            question_lstm_output, _ = tf.nn.dynamic_rnn(stack_cells, question, sequence_length=self.question_length, dtype = tf.float32)

        #PUZZLE: what will happen if there batch of answer 
        with tf.variable_scope('train',reuse=True):
            answer_lstm_ouput, _ = tf.nn.dynamic_rnn(stack_cells, answer, sequence_length=self.answer_length_list, dtype = tf.float32)
        
        ##########################################
        ### Transpose the LSTM ouput
        ##########################################
        # transpose [1, 0, 2] means the original one is [0, 1, 2] and we swap the first 2 dimensions.
        # swap the batch_size with sequence size.
        question_lstm_output = tf.transpose(question_lstm_output, [1, 0, 2])
        answer_lstm_ouput = tf.transpose(answer_lstm_ouput, [1, 0, 2])
        
        #mean pool in every cell output
        question_lstm_output = tf.reduce_mean(question_lstm_output,axis=0)
        answer_lstm_ouput = tf.reduce_mean(answer_lstm_ouput, axis=0)
        

        ##########################################
        ### Initializier User Vector
        ##########################################
        #TODO: use user profile to initialize user vector
        user = tf.get_variable("user", [self.datainfo.userSize, self.datainfo.userDim], initializer=xavier_initializer())


        ##########################################
        ### Loss Function: loss = lr + lambda * regularization
        ##########################################
        ## Lr = sum(max(0, c + f(q_i, bad_a, bad_u) - sum( f(q_j, good_a, good_u)) ))

        a_good = tf.reshape(answer_lstm_ouput[0],[128,1])
        u_good = tf.reshape(tf.gather(user, self.answer_user_placeholder[0]),[128,1])
        c = tf.constant(self.config.constant_bias)
        u = tf.gather(user, self.answer_user_placeholder)
        answer_user = tf.stack([answer_lstm_ouput, u], axis=1)
        def getloss(tensor):
            a_bad = tf.reshape(tensor[0, :], [128, 1])
            u_bad = tf.reshape(tensor[1, :], [128, 1])
            th = tf.matmul(question_lstm_output, a_bad)
            th1 = tf.matmul(question_lstm_output, u_bad)
            th2 = tf.matmul(question_lstm_output, a_good)
            th3 = tf.matmul(question_lstm_output, u_good)
            return c +th * th1  -  th2*th3

        Loss = tf.map_fn(getloss, answer_user[1::-1], dtype=tf.float32)

        
        ##########################################
        ### Evaluation: accuracy: If best answer get the hightes score => return 1;  else => return 0
        ##########################################
        compareValue = tf.zeros([1,1])
        mask = tf.less(Loss, compareValue + c)
        pred = tf.cond(tf.size(tf.boolean_mask(Loss,mask)) > tf.convert_to_tensor(0), lambda: tf.convert_to_tensor(0),lambda: tf.convert_to_tensor(1))
        Loss = tf.map_fn(lambda x: tf.maximum(compareValue, x), Loss)

        ## reg = sum( u_i - sum(vote_i_this_question / vote_all_this_question * u_j_other_answer) )
        # norm = []
        # #PUZZLE: enumerate function in tensorflow ???
        # index = 0
        # #PUZZLE: for loop in tensor
        # for user_id in self.answer_user_placeholder:
        #     index_tensor = tf.convert_to_tensor(index)
        #     u1_vote_ratio = tf.gather(self.answer_vote_placeholder, index_tensor)
        #     u1_vec = tf.gather(user, user_id)
        #     u2_matrix = tf.gather(user, self.answer_user_placeholder)
        #     norm.append( u1_vec - tf.reduce_sum(tf.multiply(u1_vote_ratio , u2_matrix )))
        #     index = index + 1
        # norm = tf.stack(norm)
        # norm = tf.reduce_sum(norm)
        error_norm = tf.reduce_sum(Loss) #+ self.config.Lambda * norm
        #accuracy evaluation

        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(error_norm)
        return pred, error_norm, train_op

    def predict_on_batch(self, sess, answer, question, answer_length_list, answer_user_list,
        answer_vote_list, question_length, state_keep_prob):
        feed = self.create_feed_dict( answer, question, answer_length_list, answer_user_list,
        answer_vote_list, question_length, state_keep_prob )

        prediction = sess.run(self.pred, feed_dict=feed)
        return prediction

    def train_on_batch(self, sess, answer, question, answer_length_list, answer_user_list,
      answer_vote_list, question_length, keep_prob=0.2):
        feed = self.create_feed_dict(
            answer, question, answer_length_list, answer_user_list,
            answer_vote_list, question_length, keep_prob
        )
        _, loss, prediction = sess.run([self.train_op, self.loss, self.pred], feed_dict=feed)
        return loss, prediction
