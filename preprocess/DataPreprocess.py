import pandas as pd 
import numpy as np 
import GetData
import gc

# 准备将用户在同一个问题下的回答当做 社交关系, 用户之间的关系类似于 pageRank 算法, 
# U_i = sum(votes_{people who answer this question} / votes_{person i get votes from this quesiton} * {people who answer this quesiton} )  
# assume accepted answer will receive 1.5 upvote than before.
'''
assume accepted answer will receive 1.5 upvote than before.
'''
def scaleVote(question, votes, scaleValue=1.5):
    accepted_ans = question.AcceptedAnswerId
    votes = votes.set_index('PostId',drop=False)
    votes[accepted_ans] = votes[accepted_ans] * scaleValue
    return votes

'''
question: Id, AcceptedAnswerId
answer: Id, OwnerUserId, ParentId
votes: PostID, VoteCount (only consider upvotes)
-----
@return: question_anwer_pair: QuestionID, GoodAnswerID, GoodUserID, BadAnswerID, BadUserID
        votes: accepted post will have 1.5 more vote than before
'''
def question_answer_user_good_bad_pair(question, answer, votes):
    answer = answer.set_index('PostId',drop=False)
    answer_vote = answer.merge(votes, how="left", left_on="Id", right_on="PostId")
    values = {"VoteCount":0}
    answer.fillna(value=values,inplace=True)
    
    # get the largest vote answer for each question 
    # method: https://stackoverflow.com/questions/15705630/python-getting-the-row-which-has-the-max-value-in-groups-using-groupby
    answer_vote = answer_vote.groupby('ParentID')['VoteCount'].transform(max) == answer_vote['VoteCount']
    answer_vote.drop("VoteCount",inplace=True,axis=1)
    question_anwer_pair = question.merge(answer_vote, how="left", left_on="Id", right_on="ParentId",rsuffix="_g_ans")
    bad_answer = answer.drop(question_anwer_pair.Id_ans,axis=0)
    question_anwer_pair = question_anwer_pair.merge(bad_answer,how="left",left_on="Id",right_on="ParentId",rsuffix="_b_ans")
    
    question_anwer_pair.drop(['AcceptedAnswerId','ParentId_g_ans','ParentId_b_ans','AcceptedAnswerId'],inplace=True, axis=1)
    question_anwer_pair.columns = ['QuestionId','GoodAnswerID','GoodUserID','BadAnswerID','BadUserID']
    return question_anwer_pair.reset_index()

'''
TODO: answer 已经包含 questionid, 所以可以不用 question 变量
question: Id, AcceptedAnswerId
answer: Id, OwnerUserId, ParentId
votes: PostId, VoteCount (only consider upvotes)
-----
return QuestionId, UserId, VoteCount
'''
def relationship(question, answer, votes):
    answer_vote = answer.merge(votes, how="left", left_on="Id", right_on="PostId")
    values = {"VoteCount":0}
    answer.fillna(value=values,inplace=True)
    question_user = question.merge(answer, how="left",left_on="Id",right_on="ParentId")
    question.columns = ['QuestionId','UserId','VoteCount']
    return question_user


def QuestionAnswerContent(rawData):
    return rawData[['Id','Body']]
    

def QuestionAnswerId(rawData):
    answer = post[post['PostTypeId'] == '2']
    question = post[post['PostTypeId'] == '1']
    return question[['Id', 'AcceptedAnswerId']],answer[['Id', 'OwnerUserId', 'ParentId']]

def VoteCount(rawData, voteType):
    votes = rawData[rawData['VoteTypeId'] == voteType]
    votes_group =   votes.groupby(['PostId'])['Id'].count()
    votes_group.reset_index()
    votes = votes.rename(index=str,columns={'index':'PostId'})
    return votes

'''
answer: Id, OwnerUserId, ParentId
votes: PostId, VoteCount (only consider upvotes)
------------
@return: answer_vote: 'QuestionId','AnswerId','UserId','VoteCount'
'''
def question_answer_user_pair(answer, votes):
    answer_vote = answer.merge(votes,how="left",left_on="Id",right_on="PostId")
    values = {"VoteCount":0}
    answer_vote.fillna(value=values,inplace=True)
    answer_vote.columns = ['AnswerId','UserId','QuestionId','VoteCount']
    answer_vote = answer[['QuestionId','AnswerId','UserId','VoteCount']]
    return answer_vote

def getScore(value_list):
    sum_ = np.sum(value_list)
    value_list = [ value * 1.0 / sum_  for value in value_list ]
    return value_list
    




