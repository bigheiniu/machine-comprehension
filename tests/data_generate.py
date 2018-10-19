from preprocess import GenerateData
import numpy as np
import pandas as pd
from Config import Config
config = Config
genter = GenerateData.DataSetLoad(False, Config.ordianry_fileName)
content, user_answer_list = genter.loadData(True,Config.pickle_fileName)
print(content.shape)
print(user_answer_list.shape)
