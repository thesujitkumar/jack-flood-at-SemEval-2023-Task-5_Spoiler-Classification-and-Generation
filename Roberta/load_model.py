from __future__ import division
from __future__ import print_function

import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import pickle
import pandas as pd


# IMPORT CONSTANTS
from models import Constants
# NEURAL NETWORK MODULES/LAYERS
from models import RoBERT, doc_cls
#from models import model_t, model_s, model_new
# DATA HANDLING CLASSES
# from models import Vocab
# # DATASET CLASS FOR SICK DATASET
# from models import Dataset
# # METRICS CLASS FOR EVALUATION
from models import Metrics
# UTILITY FUNCTIONS
from models import utils
# TRAIN AND TEST HELPER FUNCTIONS
from models import Trainer
# CONFIG PARSER
from config import parse_args

import time
import gc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def label_exteract(data_dic):    # Function to exteract label
    label_list=[]
    for idx in (range(len(data_dic.keys()))):  # exteract labels from dictionary for train data
        label = data_dic[idx]['headline']['label']
        label_list.append(label)
    target_val=torch.LongTensor(label_list)
    del label_list
    return target_val

# MAIN BLOCK
def main():
    t_start = time.time()
    global args
    args = parse_args()
    log_dir = os.path.join(args.save, args.model_name)
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    device = torch.device("cpu")
    # argument validation
    # args.cuda = args.cuda and torch.cuda.is_available()
    # device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')

    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    train_dir =   os.path.join(args.data, 'train/')#'data/sick/train/'
    #dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'Test/') #'data/sick/test/' #
    # print(train_dir, test_dir )






    args.freeze_embed= True

    if args.model_name == 'RoBERT':
	    model = RoBERT.SimilarityTreeLSTM(
		args.input_dim,
		args.mem_dim,
		args.hidden_dim,
        args.sparse,
		args.num_classes,
		args.freeze_embed,
        args.max_num_para,
        args.max_num_sent,
        args.max_num_word,
        args.domain_feature)
    elif args.model_name == 'doc_cls':
	    model = doc_cls.SimilarityTreeLSTM(
		args.input_dim,
		args.mem_dim,
		args.hidden_dim,
        args.sparse,
		args.num_classes,
		args.freeze_embed,
        args.max_num_para,
        args.max_num_sent,
        args.max_num_word,
        args.domain_feature)



    print(' Total number of parameter : {}'.format(count_parameters(model)))
    print('Number of parameters in DOCLstm:{}'.format(count_parameters(model.doclstm)))
    #print('Number of parameters in BERT Encoder :{}'.format(count_parameters(model.doclstm.Bert_encoder)))
    #print('Number of parameters in LSTM :{}'.format(count_parameters(model.doclstm.News_LSTM)))

    # print( 'Size of model : {} byte '.format(sys.getsizeof(model)))


    print(' Total number of parameter : {}'.format(count_parameters(model)))
    print('Number of parameters in DOCLstm:{}'.format(count_parameters(model.doclstm)))
   # print('Number of parameters in Tree-lstm :{}'.format(count_parameters(model.doclstm.childsummodels)))
    print( 'Size of model : {} byte '.format(sys.getsizeof(model)))

    criterion =  nn.CrossEntropyLoss() #nn.KLDivLoss() #   nn.CrossEntropyLoss()

    #NELA_Data_GLOV_EMBED_200d.pth
    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors

    # delete emb
    model.to(device), criterion.to(device)
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # exteract labels for training data
    trainer = Trainer(args, model, criterion, optimizer, device, args.batchsize, args.num_classes, args.file_len, args.domain_feature)

    checkpoint_path  = '{}.pt'.format(os.path.join(log_dir, args.expname))
    # checkpoint_path  = '{}.pt'.format(os.path.join('checkpoints_back', args.expname))
    checkpoint= torch.load(checkpoint_path)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model.eval()
    trainer.optimizer.load_state_dict(checkpoint['optim_state_dict'])
    #trainer.optimizer.eval()
    test_loss, test_pred = trainer.test(2) # make it 2
    print(test_pred)
    #test_pred = trainer.test(2)
    print("test predictions hape",test_pred.size())
    fname = os.path.join(test_dir, 'test_label.pkl') # Test label .pkl file name
    fin = open(fname , 'rb')
    ground_truth = pickle.load(fin)
    print("the length of total label:",len(ground_truth))
    fin.close()
    test_accuracy = metrics.accuracy(test_pred, ground_truth)
    test_fmeasure = metrics.fmeasure(test_pred, ground_truth)
    logger.info(' test \tLoss: {}\tAccuracy: {}\tF1-score: {}'.format(
                 test_loss, test_accuracy , test_fmeasure))
    df=pd.DataFrame(test_pred)
    df.to_csv("ans.csv",index=False)
    #del ground_truth


if __name__ =='__main__':
    main()
