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
    ordianry_fileName = []
    pickle_fileName = []
    loadPickle = False
    isStore = True
    valueCountSmooth = 0.0

class DataInfo:
    userDim = 200
    
    def __init__:(userSize):
        self._usersize = userSize
    @property
    def userSize():
        return self._usersize

    
