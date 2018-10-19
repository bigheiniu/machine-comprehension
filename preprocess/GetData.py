from preprocess.ParseXML import XML2DF
import pandas as pd 
import pickle
'''
https://ia600107.us.archive.org/27/items/stackexchange/readme.txt
'''
def getPostData(fileName):
    # fileName = "/Users/bigheiniu/course/ASU_course/472_social/classproject/stackoverflow/data/Posts.xml"
    columns = ['Id','PostTypeId','Body','ParentId','AcceptedAnswerId','OwnerUserId']
    data = XML2DF(fileName,columns)
    return data


def getVotesRelationship(fileName):
    columns = ['PostId','VoteTypeId']
    data = XML2DF(fileName, columns)
    data
    return data

