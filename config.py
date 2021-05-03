class Config():
    DATA_DIR = 'data'
    SAVE_DIR = 'save'
    
    seed = 42
    epochs = 300
    num_wifi_feats = 20
    fold_num = 5
    train_batch_size = 256
    val_batch_size = 256
    num_workers = 16
    device = 'gpu'
    neptune = False
    lr = 5e-3


