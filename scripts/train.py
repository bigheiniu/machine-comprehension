import sys
import random
import numpy as np
from tqdm import tqdm
import argparse
import tensorflow as tf
from multiprocessing import Pool
from sklearn.externals import joblib
from os.path import abspath, dirname, join, exists
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from model import DeepLSTM
from preprocess.GenerateData import DataSetLoad 
import gc


class Config:
    vocab_size = 100002
    embed_size = 200
    state_size = 256
    depth = 2
    lr = 5E-4
    batch_size = 32
    num_sampled = 512
    n_epoch = 512
    train_steps = 200


class Train:
    def __init__(self, config, debug=True):
        self.config = config
        self.debug = debug


    def load_data(self):
        dataloader = DataSetLoad(self.config.loadPicke, self.config.ordinary_fileName)
        content, question_user_vote = dataloader(self.config.isStore, self.config.pickle_fileName)
        return content, question_user_vote

    #one question -> multiple answer
    def gen_data(self, content, question_user_vote_group, questionId):
        m = question_user_vote_group.get_group(questionId)
        sum_vote = np.sum(m['VoteCount']) + self.config.valueCountSmooth
        m['VoteCount'] = m['VoteCount'].apply(lambda x: x *1.0 / sum_vote)
        sorted_m = m.sort_values(by=['VoteCount'])
        
        answer_id_list = sorted_m['AnswerId']
        user_id_list = sorted_m['UserId']
        answer_vote_list = sorted_m['VoteCount']
        question_content = content[questionId,'Body']
        answer_content_list = contnet[answer_id,'Body']
        question_length = question_content.apply(len)
        answer_length_list = answer_content.apply(len)
        return answer_content_list, question_content, answer_length_list, \
        user_id_list, answer_vote_list, question_length




        

    def train(self, mc_model, model_output):
        # if self.debug:
        #     data_set = self.validation_set
        #     random.seed(10)
        # else:
        content, question_user_vote = self.load_data()
        question_user_vote_group = question_user_vote.groupby('QuestionId')
        gc.enable()
        del question_user_vote
        gc.collect()
        questionIds = question_user_vote_group.groups
        
        with tf.Graph().as_default():
            max_f1_score = 0.0
            model = mc_model(self.config)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as session:
                session.run(init)
                session.graph.finalize()
                for epoch in range(self.config.n_epoch):
                    pbar = tqdm(range(self.config.train_steps), desc="{} epoch".format(epoch))
                    avg_loss, avg_acc = 0.0, 0.0
                    for _ in pbar:
                        inputs, length, labels = self.gen_file(data_set)
                        loss, prediction = model.train_on_batch(session, inputs, length, labels)
                        mean_loss = np.mean(loss)
                        acc = accuracy_score(labels, prediction)
                        avg_loss += mean_loss
                        avg_acc += acc
                        pbar.set_description("loss/acc: {:.2f}/{:.2f}".format(mean_loss, acc))
                    print("avg_loss/avg_acc: {:.2f}/{:.2f} on this epoch".format(avg_loss/self.config.train_steps, avg_acc/self.config.train_steps))
                    inputs, length, labels = self.gen_file(self.test_set, False)
                    prediction = model.predict_on_batch(session, inputs, length)
                    f1 = f1_score(labels, prediction, average='micro')
                    print("evaluate: acc/pre/rec/f1: {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
                        accuracy_score(labels, prediction),
                        precision_score(labels, prediction, average='micro'),
                        recall_score(labels, prediction, average='micro'),
                        f1
                    ))
                    if f1 > max_f1_score:
                        print("New best score! Saving model in {}".format(model_output))
                        max_f1_score = f1
                        saver.save(session, model_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", help="数据集的 question 目录")
    parser.add_argument("-p", "--pickle",help="文件类型是pickle")
    args = parser.parse_args()
    config = Config()
    train = Train(args.root, config, debug=False)
    train.train(mc_model=DeepLSTM, model_output=join(args.root, "model.ckpt"))
