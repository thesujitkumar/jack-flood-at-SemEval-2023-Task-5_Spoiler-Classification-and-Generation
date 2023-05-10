##  model with simle LSTM for sentence encoding (without tree lstm)

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from . import Constants






class DocLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word):
        super(DocLSTM, self).__init__()

        self.max_num_sent = max_num_sent

        self.mem_dim = mem_dim
        self.in_dim = in_dim

        # self.proj= nn.Linear(self.in_dim , 2*self.mem_dim)
        self.News_LSTM = self.sentence_BILSTM = nn.LSTM(in_dim, mem_dim, 1,bidirectional=True)

        torch.manual_seed(0)
        self.sent_pad =   torch.randn(1, in_dim)


    def forward(self, body):

        rsent = body['headline']['rsent']
        # head= self.proj(rsent)



        body=body['body_list']
        count=0
        sent_encoded_List= []
        sent_encoded_List.append(rsent.view(1,768))
        for p_id in body:
            for s_id, sentence in enumerate(body[p_id]):

                lsent = sentence
                sent_encoded_List.append(lsent.view(1,768))
        "encoding of each segmente on top of encdoings of each segments using BERT"

        sent_encoded_List += [ self.sent_pad] * (self.max_num_sent - len(self.sent_pad))
        news_article_inp = torch.cat(sent_encoded_List[:self.max_num_sent], 0)
        # print("len of text",len(sent_encoded_List))
        del sent_encoded_List

        out_News_article, (h_News_article, c_News_article)=self.News_LSTM(news_article_inp.contiguous().view(self.max_num_sent, 1, self.in_dim))
        body_hid_2d=h_News_article.view(2,100)
        body_sent_left=body_hid_2d[0]
        body_sent_right=body_hid_2d[1]
        Bi_body_sent_h=torch.cat((body_sent_left,body_sent_right),0)

        # exit(1)



        del out_News_article, c_News_article, body, rsent, lsent
        # return head,Bi_body_sent_h
        return Bi_body_sent_h




# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes, domain_feature):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.domain_feature= domain_feature


        if self.domain_feature : # contact deep feature + domain feature
            self.wh = nn.Linear(((8*self.mem_dim) + self.feature_dim), self.hidden_dim) # for combined model
        else: # use only deep feature
            self.wh = nn.Linear((2 * self.mem_dim) , self.hidden_dim)  # for only deep feature.
        self.wp = nn.Linear(self.mem_dim,3)

    def forward(self,body,feature_vec):
        # mult_dist = torch.mul(head, body) #dot product between body and headline representation
        # abs_dist = torch.abs(torch.add(head, -body))  # absoulte difference between body and headline representation
        # vec_dist = torch.cat((mult_dist, abs_dist), 1)  # concatenation of absoulte difference and multiplication
        # vec_cat=torch.cat((head,body),1) # concatenation of body and headline vectors
        # entail=torch.cat((vec_dist,vec_cat),1)

        # """ Merge the feature vecot befor going to MLP"""
        if self.domain_feature: #for combined feature model
            concat_vec = torch.cat( (body , torch.FloatTensor(feature_vec).reshape(1,len(feature_vec)) ), dim=1)
            #print(' concst vec shape : ', concat_vec.shape)
            out = torch.sigmoid(self.wh(concat_vec)) # Calling MLP for combined model
        else:
            out = torch.sigmoid(self.wh(body)) # for model with only deep feature
            out =self.wp(out) # No softmax
        #print(out)
        return out




# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self,  in_dim, mem_dim, hidden_dim, sparsity, freeze, num_classes, \
        max_num_para, max_num_sent, max_num_word,domain_feature):
        super(SimilarityTreeLSTM, self).__init__()
        self.mem_dim = mem_dim
        self.doclstm = DocLSTM( in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word)
        self.similarity = Similarity( mem_dim, hidden_dim, num_classes,domain_feature)
    def forward(self, body,  feature_vec = None):
        News_encoding  = self.doclstm(body)
        output = self.similarity(News_encoding.view(1, 2*self.mem_dim),feature_vec )
        return output
