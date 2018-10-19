class Config:
    # vocab_size = 100002
    vocab_size = 10002
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
    ordianry_fileName = ["/home/bigheiniu/course/ASU_Course/472/coursePro/machine-comprehension/resource/Votes.xml","/home/bigheiniu/course/ASU_Course/472/coursePro/machine-comprehension/resource/Posts.xml"]
    pickle_fileName = ["/home/bigheiniu/course/ASU_Course/472/coursePro/machine-comprehension/resource/content.pickle","/home/bigheiniu/course/ASU_Course/472/coursePro/machine-comprehension/resource/question_answer_vote.pickle"]
    loadPickle = True
    isStore = False
    valueCountSmooth = 0.1
    word2vect_dir = "/home/bigheiniu/course/ASU_Course/472/coursePro/machine-comprehension/resource/GoogleNews-vectors-negative300.bin.gz"
    #if this answer be accepted, it will have 1.5 more votes
    voteScale = 1.5
    VoteTypeId = "2"
    constant_bias = 0.1
    Lambda = 1

class DataInfo:
    userSize = 1000000 # greater than actual size
    userDim = 128

    

