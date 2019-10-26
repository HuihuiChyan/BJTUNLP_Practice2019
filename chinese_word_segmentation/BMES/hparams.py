class Hparams:
    train_path = './data/msr_training.utf8'
    test_gold_path = './data/msr_test_gold.utf8'
    prepro_dir = './data/prepro/'
    vocab_path = './data/msr.vocab'
    result_path = './results/'
    vocab_size = 6000
    batch_size = 64
    max_len = 800
    tag2label = {"B": 0,
                 "M": 1,
                 "E": 2,
                 "S": 3,
                 }


    # model
    num_units = 512

    # train
    lr = 0.001
    max_to_keep = 5
    log_dir = './logs/'
    epoch = 10
    steps_per_save = 10000
