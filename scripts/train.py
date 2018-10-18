import sys
import random
import numpy as np
from tqdm import tqdm
import argparse
import tensorflow as tf
from os.path import abspath, dirname, join, exists
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from model.deep_lstm import DeepLSTM
from preprocess.GenerateData import DataSetLoad 
from preprocess.convert2index import Word2index
import Config
import gc
from utils import padding_easy




class Train:
    def __init__(self, config, datainfo, debug=True):
        self.config = config
        self.debug = debug
        self.datainfo = datainfo


    def load_data(self):
        dataloader = DataSetLoad(self.config.loadPicke, self.config.ordinary_fileName)
        content, question_user_vote = dataloader.loadData(self.config.isStore, self.config.pickle_fileName)
        return content, question_user_vote

    #one question -> multiple answer
    def gen_data(self, content, question_user_vote_group, questionId):
        m = question_user_vote_group.get_group(questionId)
        sum_vote = np.sum(m['VoteCount']) + self.config.valueCountSmooth
        # ratio of vote 
        m['VoteCount'] = m['VoteCount'].apply(lambda x: x * 1.0 / sum_vote)
        ##all transported from pandas to numpy
        # TODO: performance--use List
        sorted_m = m.sort_values(by=['VoteCount'])
        answer_id_list = sorted_m['AnswerId'].values
        user_id_list = sorted_m['UserId'].values
        answer_vote_list = sorted_m['VoteCount'].values
        question_content = content[questionId,'Body']
        # padding to have the same length
        question_content = padding_easy(question_content.values, self.config.maxlen)
        answer_content_list = content[answer_id_list,'Body']
        answer_content_list = padding_easy(answer_content_list.values, self.config.maxlen)

        question_length = question_content.apply(len)
        answer_length_list = answer_content_list.apply(len)
        return answer_content_list, question_content, answer_length_list, \
        user_id_list, answer_vote_list, question_length

    def train(self, mc_model, model_output):
        # if self.debug:
        #     data_set = self.validation_set
        #     random.seed(10)
        # else:
        content, question_user_vote = self.load_data()
        question_user_vote_group = question_user_vote.groupby('QuestionId')

        word2index =  Word2index(content["Body"].values)
        # convert word into index
        content["Body"] = word2index.convert(content["Body"])
        embedMatrix = word2index.loadEmbeddingMatrix(self.config.word2vect_dir)
        # load 
        gc.enable()
        del question_user_vote
        gc.collect()
        questionIds_list = question_user_vote_group.groups
        question_size = len(questionIds_list)
        with tf.Graph().as_default():
            max_f1_score = 0.0
            model = mc_model(self.config, self.datainfo, embedMatrix)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as session:
                session.run(init)
                session.graph.finalize()
                for epoch in range(self.config.n_epoch):
                    pbar = tqdm(range(question_size), desc="{} epoch".format(epoch))
                    avg_loss, avg_acc = 0.0, 0.0
                    flag = 0
                    for question_id in questionIds_list:
                        answer_content_list, question_content, answer_length_list, \
        user_id_list, answer_vote_list, question_length = self.gen_data(content, question_user_vote_group, question_id)
                        loss, prediction = model.train_on_batch(session, answer_content_list, question_content, answer_length_list, \
                        user_id_list, answer_vote_list, question_length)
                        acc = acc + prediction
                        avg_loss += loss
                        flag = flag + 1
                        acc_ratio = acc * 1.0 / (flag * 1.0)
                        pbar.set_description("loss/acc: {:.2f}/{:.2f}".format(avg_loss, acc_ratio))
                        
                    # prediction = model.predict_on_batch(session, inputs, length)
                    # f1 = f1_score(labels, prediction, average='micro')
                    # print("evaluate: acc/pre/rec/f1: {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
                    #         accuracy_score(labels, prediction),
                    #         precision_score(labels, prediction, average='micro'),
                    #         recall_score(labels, prediction, average='micro'),
                    #         f1
                    #     ))
                    # if f1 > max_f1_score:
                    #     print("New best score! Saving model in {}".format(model_output))
                    #     max_f1_score = f1
                    #     saver.save(session, model_output)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-r", "--root", help="数据集的 question 目录")
    # parser.add_argument("-p", "--pickle",help="文件类型是pickle")
    # args = parser.parse_args()
    # config = Config()
    config = Config.Config
    datainfo = Config.DataInfo
    train = Train(config, datainfo, debug=False)
    train.train(mc_model=DeepLSTM, model_output="model.ckpt")
