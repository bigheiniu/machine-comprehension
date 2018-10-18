import numpy as np 
import pandas as pd 
import pickle
from convert2index import Word2index

class DataSetLoad:
    
    '''
    #fileName: 
            loadPickle - True => fileName[0]->"Id","Body" 
                                 fileName[1]->'QuestionId','AnswerId','UserId','VoteCount'
            loadPickle - False => fileName[0]->Votes.xml
                                  fileName[1]->Post.xml
    '''

    def __init__(self, loadPickle, fileName):
        self.loadPickle = loadPickle
        self.fileName = fileName
    
    def loadData(self, isStore=False, StoreLoaction=None):
        if(self.loadPickle):
            with open(fileName[0], "rb") as f1:
                content = pickle.load(f1)
            with open(fileName[1],"rb") as f1:
                question_user_vote = pickle.load(f1)
        else:
            votes = GetData.getVotesRelationship(fileName[0],isStore, StoreLoaction)
            post = GetData.getPostData(fileName[1], isStore, StoreLoaction)
            content = DataPreprocess.QuestionAnswerContent(post)
            question,answer = DataPreprocess.QuestionAnswerId(post)
            votes = DataPreprocess.scaleVote(question,votes)
            question_user_vote = DataPreprocess.relationship(question, answer, votes)
            # clean text
            content.iloc[:,0] = TextDataPreprocess.text_to_wordlist(content.iloc[:,0])
            if(isStore):
                with open(StoreLoaction[0],'wb') as f1:
                    pickle.dump(f1, content)
                with open(StoreLoaction[1],'wb') as f1:
                    pickle.dump(f1,  question_user_vote)
        return content, question_user_vote

    def LoadEmbedding(self, corpus):
        self.word2indx = Word2index(corpus)
        
    
                    

        