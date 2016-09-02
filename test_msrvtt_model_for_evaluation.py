#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import ipdb

import cv2

from tensorflow.python.ops import rnn_cell
from keras.preprocessing import sequence

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, drop_out_rate, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.drop_out_rate = drop_out_rate

        with tf.device("/gpu:3"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

#         self.lstm1 = rnn_cell.BasicLSTMCell(dim_hidden)
#         self.lstm2 = rnn_cell.BasicLSTMCell(dim_hidden)
        
        self.lstm1 = rnn_cell.LSTMCell(self.dim_hidden,self.dim_hidden,use_peepholes = True)
        self.lstm1_dropout = rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1 - self.drop_out_rate)
        self.lstm2 = rnn_cell.LSTMCell(self.dim_hidden,self.dim_hidden,use_peepholes = True)
        self.lstm2_dropout = rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)
    
        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')


    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_lstm_steps, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_lstm_steps])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(self.n_lstm_steps):
            if i > 0: tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                #output1, state1 = self.lstm1( image_emb[:,i,:], state1 )
                output1, state1 = self.lstm1_dropout( image_emb[:,i,:], state1 )
                
            with tf.variable_scope("LSTM2"):
                #output2, state2 = self.lstm2( tf.concat(1,[padding,output1]), state2 )
                output2, state2 = self.lstm2_dropout( tf.concat(1,[padding,output1]), state2 )

        for i in range(self.n_lstm_steps):

            tf.get_variable_scope().reuse_variables()

            if i == 0:
                current_embed = tf.zeros([1, self.dim_hidden])

            with tf.variable_scope("LSTM1"):
                #output1, state1 = self.lstm1( padding, state1 )
                output1, state1 = self.lstm1_dropout( padding, state1 )
                
            with tf.variable_scope("LSTM2"):
                #output2, state2 = self.lstm2( tf.concat(1,[current_embed,output1]), state2 )
                output2, state2 = self.lstm2_dropout( tf.concat(1,[current_embed,output1]), state2 )

                
            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/gpu:3"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds


############### Global Parameters ###############
video_path = './youtube_videos'
video_data_path='./video_corpus.csv'
video_feat_path = './youtube_feats'


############## Train Parameters #################
dim_image = 4096
dim_hidden= 256
n_frame_step = 80
n_epochs = 1000
batch_size = 100
learning_rate = 0.001
##################################################

def get_video_data(video_data_path, video_feat_path, train_ratio=0.9):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = video_data['video_path'].unique()
    train_len = int(len(unique_filenames)*train_ratio)

    train_vids = unique_filenames[:train_len]
    test_vids = unique_filenames[train_len:]

    train_data = video_data[video_data['video_path'].map(lambda x: x in train_vids)]
    test_data = video_data[video_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, test_data

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector


def test(model_path, test_feats_path):
    test_videos = []
    #for MSVD evaluation
    MSVDtest_range = range(1301,1971) #1301~1970
    for i in MSVDtest_range:    
        test_videos.append( os.path.join(test_feats_path, 'vid'+str(i)+'.avi.npy') )

#     MSRVTTval_range = range(6513,7010) #6513~7009
#     for i in MSRVTTval_range:    
#         test_videos.append( os.path.join(test_feats_path, 'video'+str(i)+'.mp4.npy') )


    ixtoword = pd.Series(np.load('./data/MSRVTT_ot/ixtoword.npy').tolist())
    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            drop_out_rate = 0,
            bias_init_vector=None)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    output_file = './results/MSVD_test_vgg16_msrvtt3_900.txt'     
    #output_file = './results/MSRVTT_val_vgg16_msrvtt3_700.txt' 
    with open(output_file, 'w') as f:
        f.write('')
    
    
    for video_feat_path in test_videos:
        print video_feat_path
        video_feat = np.load(video_feat_path)[None,...]
        if video_feat.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

        else:
            shape_templete = np.zeros(shape=(1, n_frame_step, 4096), dtype=float )
            shape_templete[:video_feat.shape[0],:video_feat.shape[1],:video_feat.shape[2]] = video_feat
            video_feat = shape_templete
            video_mask = np.ones((video_feat.shape[0], n_frame_step))

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        probs_val = sess.run(probs_tf, feed_dict={video_tf:video_feat})
        embed_val = sess.run(last_embed_tf, feed_dict={video_tf:video_feat})
        generated_words = ixtoword[generated_word_index]

        punctuation = np.argmax(np.array(generated_words) == '.')+1
        generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        video_id = os.path.basename(video_feat_path).split('.')[0].replace('video', 'vid')
        with open(output_file, 'a') as f:
            f.write( '\t{}\t{}\n'.format( video_id, generated_sentence ) )
            
        print '{} {}'.format( video_id, generated_sentence )


test('MSRVTT_models3/model-900', '/home2/dataset/MSVD/MSVD_test_feats')
#test('MSRVTT_models3/model-700', '/home2/dataset/MSR-VTT/train_val_feats')
