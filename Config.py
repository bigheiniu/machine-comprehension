class Config:
    vocab_size = 100002
    embed_size = 200
    state_size = 256
    depth = 1
    numhidden = 128
    lr = 5E-4
    batch_size = 32
    num_sampled = 512
    n_epoch = 512
    train_steps = 200
    maxlen = 200
    ordianry_fileName = ["resource/Posts.xml", "resource/Votes.xml"]
    pickle_fileName = ["resource/content.pickle","resource/question_answer_vote.pickle"]
    loadPickle = False
    isStore = True
    valueCountSmooth = 0.1
    word2vect_dir = "resource/GoogleNews-vectors-negative300.bin.gz"
    #if this answer be accepted, it will have 1.5 more votes
    voteScale = 1.5

class DataInfo:
    userDim = 200
    userSize = 2000 # greater than actual size
    userDim = 128

    

