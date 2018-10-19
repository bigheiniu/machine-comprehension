import numpy as np 
import pandas as pd 
import pickle
from preprocess import GetData, DataPreprocess, TextDataPreprocess, convert2index
from Config import Config

class DataSetLoad:
    
    '''
    #fileName: 
            loadPickle - True => fileName[0]->"Id","Body" 
                                 fileName[1]->'QuestionId','AnswerId','UserId','VoteCount'
            loadPickle - False => fileName[0]->Votes.xml
                                  fileName[1]->Post.xml
    '''

    def __init__(self, ):
        self.loadPickle = Config.loadPickle
    
    def loadData(self):
        if self.loadPickle :
            print("load pickable file")
            with open(Config.pickle_fileName[0], "rb") as f1:
                content = pickle.load(f1)
            with open(Config.pickle_fileName[1],"rb") as f1:
                question_user_vote = pickle.load(f1)
        else:
            votes = GetData.getVotesRelationship(Config.ordianry_fileName[0])
            votes = DataPreprocess.VoteCount(votes, Config.VoteTypeId)
            post = GetData.getPostData(Config.ordianry_fileName[1])
            content = DataPreprocess.QuestionAnswerContent(post)
            question, answer = DataPreprocess.QuestionAnswerId(post)
            votes = DataPreprocess.scaleVote(question,votes)
            question_user_vote = DataPreprocess.relationship(question, answer, votes)
            # clean text
            content.loc[:, "Body"] = TextDataPreprocess.text_to_wordlist(content.loc[:, "Body"])
            if(Config.isStore):
                with open(Config.pickle_fileName[0],'wb') as f1:
                    pickle.dump(content, f1)
                with open(Config.pickle_fileName[1],'wb') as f1:
                    pickle.dump(question_user_vote, f1)
        return content, question_user_vote

    # def LoadEmbedding(self):
    #     return self.word2indx.loadEmbeddingMatrix(Config.Config.word2vect_dir)
    # def sequence2indx(self,text):
    #     return self.word2indx.convert(text)
    #
    # def setWord2index(self,corpus):
    #     self.word2indx = convert2index.Word2index(corpus)



        


        