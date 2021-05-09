
class Config(object):
    def __init__(self, args):
        self.args = args

        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.n_epoch = args.n_epoch
        self.max_seq_len = args.max_seq_len
        self.rel_num = args.rel_num

        self.dataset = args.dataset
        self.multi_gpu = args.multi_gpu

        # path
        self.data_path = './data/' + self.dataset
        self.checkpoint_dir = './checkpoint/' + self.dataset
        self.log_dir = './log/' + self.dataset
        self.result_dir = './result/' + self.dataset
        self.train_triples = args.train_triples
        self.dev_triples = args.dev_triples
        self.test_triples = args.test_triples
        self.model_save_name = args.model_name + '_dataset_' + self.dataset + '_lr_' + str(self.learning_rate) + '_batch_' + str(self.batch_size)
        self.log_save_name = 'log_' + args.model_name + '_dataset_' + self.dataset + '_lr_' + str(self.learning_rate) + '_batch_' + str(self.batch_size)
        self.result_save_name = 'result_' + args.model_name + '_dataset_' + self.dataset + '_lr_' + str(self.learning_rate) + '_batch_' + str(self.batch_size) + '.txt'

        # log setting
        self.period = args.period
        self.test_epoch = args.test_epoch

