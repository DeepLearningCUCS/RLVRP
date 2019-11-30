#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhixiang Hu (zh2366), Yunke Gan (yg2631)

@reference: Nazari, M., Oroojlooy, A., Snyder, L. V., and Taka ́c, M. Reinforcement learning for solving the vehiclerouting problem.
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import tensorflow as tf
import time
from collections import namedtuple
import sys
from datetime import datetime
import warnings
import collections
import pandas as pd

print_grad = True

###############################################################################
def create_VRP_dataset(
        n_problems,
        n_cust,
        data_dir,
        seed=None,
        data_type='train'):
    '''
    This function creates VRP instances and saves them on disk. If a file is already available,
    it will load the file.
    Input:
        n_problems: number of problems to generate.
        n_cust: number of customers in the problem.
        data_dir: the directory to save or load the file.
        seed: random seed for generating the data.
        data_type: the purpose for generating the data. It can be 'train', 'val', or any string.
    output:
        data: a numpy array with shape [n_problems x (n_cust+1) x 3]
        in the last dimension, we have x,y,demand for customers. The last node is for depot and 
        it has demand 0.
     '''

    # set random number generator
    n_nodes = n_cust +1
    if seed == None:
        rnd = np.random
    else:
        rnd = np.random.RandomState(seed)
    
    # build task name and datafiles
    task_name = 'vrp-size-{}-len-{}-{}.txt'.format(n_problems, n_nodes,data_type)
    fname = os.path.join(data_dir, task_name)

    # cteate/load data
    if os.path.exists(fname):
        print('Loading dataset for {}...'.format(task_name))
        data = np.loadtxt(fname,delimiter=' ')
        data = data.reshape(-1, n_nodes,3)
    else:
        print('Creating dataset for {}...'.format(task_name))
        # Generate a training set of size n_problems 
        x = rnd.uniform(0,1,size=(n_problems,n_nodes,2))
        d = rnd.randint(1,10,[n_problems,n_nodes,1])
        d[:,-1]=0 # demand of depot
        data = np.concatenate([x,d],2)
        np.savetxt(fname, data.reshape(-1, n_nodes*3))

    return data

class DataGenerator(object):
    def __init__(self, 
                 args):

        '''
        This class generates VRP problems for training and test
        Inputs:
            args: the parameter dictionary. It should include:
                args['random_seed']: random seed
                args['test_size']: number of problems to test
                args['n_nodes']: number of nodes
                args['n_cust']: number of customers
                args['batch_size']: batchsize for training

        '''
        self.args = args
        self.rnd = np.random.RandomState(seed= args['random_seed'])
        print('Created train iterator.')

        # create test data
        self.n_problems = args['test_size']
        self.test_data = create_VRP_dataset(self.n_problems,args['n_cust'],'./data',
            seed = args['random_seed']+1,data_type='test')

        self.reset()

    def reset(self):
        self.count = 0

    def get_train_next(self):
        '''
        Get next batch of problems for training
        Retuens:
            input_data: data with shape [batch_size x max_time x 3]
        '''

        input_pnt = self.rnd.uniform(0,1,
            size=(self.args['batch_size'],self.args['n_nodes'],2))

        demand = self.rnd.randint(1,10,[self.args['batch_size'],self.args['n_nodes']])
        demand[:,-1]=0 # demand of depot

        input_data = np.concatenate([input_pnt,np.expand_dims(demand,2)],2)

        return input_data

 
    def get_test_next(self):
        '''
        Get next batch of problems for testing
        '''
        if self.count<self.args['test_size']:
            input_pnt = self.test_data[self.count:self.count+1]
            self.count +=1
        else:
            warnings.warn("The test iterator reset.") 
            self.count = 0
            input_pnt = self.test_data[self.count:self.count+1]
            self.count +=1

        return input_pnt

    def get_test_all(self):
        '''
        Get all test problems
        '''
        return self.test_data
    

class State(collections.namedtuple("State",
                                        ("load",
                                         "demand",
                                         'd_sat',
                                         "mask"))):
    pass
    
class Env(object):
    def __init__(self,
                 args):
        '''
        This is the environment for VRP.
        Inputs: 
            args: the parameter dictionary. It should include:
                args['n_nodes']: number of nodes in VRP
                args['n_custs']: number of customers in VRP
                args['input_dim']: dimension of the problem which is 2
        '''
        self.capacity = args['capacity']
        self.n_nodes = args['n_nodes']
        self.n_cust = args['n_cust']
        self.input_dim = args['input_dim']
        self.input_data = tf.placeholder(tf.float32,\
            shape=[None,self.n_nodes,self.input_dim])

        self.input_pnt = self.input_data[:,:,:2]
        self.demand = self.input_data[:,:,-1]
        self.batch_size = tf.shape(self.input_pnt)[0] 
        
    def reset(self,beam_width=1):
        '''
        Resets the environment. This environment might be used with different decoders. 
        In case of using with beam-search decoder, we need to have to increase
        the rows of the mask by a factor of beam_width.
        '''

        # dimensions
        self.beam_width = beam_width
        self.batch_beam = self.batch_size * beam_width

        self.input_pnt = self.input_data[:,:,:2]
        self.demand = self.input_data[:,:,-1]

        # modify the self.input_pnt and self.demand for beam search decoder
#         self.input_pnt = tf.tile(self.input_pnt, [self.beam_width,1,1])

        # demand: [batch_size * beam_width, max_time]
        # demand[i] = demand[i+batchsize]
        self.demand = tf.tile(self.demand, [self.beam_width,1])

        # load: [batch_size * beam_width]
        self.load = tf.ones([self.batch_beam])*self.capacity

        # create mask
        self.mask = tf.zeros([self.batch_size*beam_width,self.n_nodes],
                dtype=tf.float32)

        # update mask -- mask if customer demand is 0 and depot
        self.mask = tf.concat([tf.cast(tf.equal(self.demand,0), tf.float32)[:,:-1],
            tf.ones([self.batch_beam,1])],1)

        state = State(load=self.load,
                    demand = self.demand,
                    d_sat = tf.zeros([self.batch_beam,self.n_nodes]),
                    mask = self.mask )

        return state

    def step(self,
             idx,
             beam_parent=None):
        '''
        runs one step of the environment and updates demands, loads and masks
        '''

        # if the environment is used in beam search decoder
        if beam_parent is not None:
            # BatchBeamSeq: [batch_size*beam_width x 1]
            # [0,1,2,3,...,127,0,1,...],
            batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(self.batch_size), tf.int64),
                                                 [self.beam_width]),1)
            # batchedBeamIdx:[batch_size*beam_width]
            batchedBeamIdx= batchBeamSeq + tf.cast(self.batch_size,tf.int64)*beam_parent
            # demand:[batch_size*beam_width x sourceL]
            self.demand= tf.gather_nd(self.demand,batchedBeamIdx)
            #load:[batch_size*beam_width]
            self.load = tf.gather_nd(self.load,batchedBeamIdx)
            #MASK:[batch_size*beam_width x sourceL]
            self.mask = tf.gather_nd(self.mask,batchedBeamIdx)


        BatchSequence = tf.expand_dims(tf.cast(tf.range(self.batch_beam), tf.int64), 1)
        batched_idx = tf.concat([BatchSequence,idx],1)

        # how much the demand is satisfied
        d_sat = tf.minimum(tf.gather_nd(self.demand,batched_idx), self.load)

        # update the demand
        d_scatter = tf.scatter_nd(batched_idx, d_sat, tf.cast(tf.shape(self.demand),tf.int64))
        self.demand = tf.subtract(self.demand, d_scatter)

        # update load
        self.load -= d_sat

        # refill the truck -- idx: [10,9,10] -> load_flag: [1 0 1]
        load_flag = tf.squeeze(tf.cast(tf.equal(idx,self.n_cust),tf.float32),1)
        self.load = tf.multiply(self.load,1-load_flag) + load_flag *self.capacity

        # mask for customers with zero demand
        self.mask = tf.concat([tf.cast(tf.equal(self.demand,0), tf.float32)[:,:-1],
                                          tf.zeros([self.batch_beam,1])],1)

        # mask if load= 0 
        # mask if in depot and there is still a demand

        self.mask += tf.concat( [tf.tile(tf.expand_dims(tf.cast(tf.equal(self.load,0),
            tf.float32),1), [1,self.n_cust]),                      
            tf.expand_dims(tf.multiply(tf.cast(tf.greater(tf.reduce_sum(self.demand,1),0),tf.float32),
                             tf.squeeze( tf.cast(tf.equal(idx,self.n_cust),tf.float32))),1)],1)

        state = State(load=self.load,
                    demand = self.demand,
                    d_sat = d_sat,
                    mask = self.mask )

        return state

def reward_func(sample_solution):
    """The reward for the VRP task is defined as the 
    negative value of the route length

    Args:
        sample_solution : a list tensor of size decode_len of shape [batch_size x input_dim]
        demands satisfied: a list tensor of size decode_len of shape [batch_size]

    Returns:
        rewards: tensor of size [batch_size]

    Example:
        sample_solution = [[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]]
        sourceL = 3
        batch_size = 2
        input_dim = 2
        sample_solution_tilted[ [[5,5]
                                                    #  [6,6]]
                                                    # [[1,1]
                                                    #  [2,2]]
                                                    # [[3,3]
                                                    #  [4,4]] ]
    """
    # make init_solution of shape [sourceL x batch_size x input_dim]


    # make sample_solution of shape [sourceL x batch_size x input_dim]
    sample_solution = tf.stack(sample_solution,0)

    sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1],0),
         sample_solution[:-1]),0)
    # get the reward based on the route lengths


    route_lens_decoded = tf.reduce_sum(tf.pow(tf.reduce_sum(tf.pow(\
        (sample_solution_tilted - sample_solution) ,2), 2) , .5), 0)
    return route_lens_decoded

###############################################################################
class AttentionVRPActor(object):
    """A generic attention module for the attention in vrp model"""
    def __init__(self, dim, use_tanh=False, C=10,_name='Attention',_scope=''):
        self.use_tanh = use_tanh
        self._scope = _scope

        with tf.variable_scope(_scope+_name):
            # self.v: is a variable with shape [1 x dim]
            self.v = tf.get_variable('v',[1,dim],
                       initializer=tf.contrib.layers.xavier_initializer())
            self.v = tf.expand_dims(self.v,2)
            
        self.emb_d = tf.layers.Conv1D(dim,1,_scope=_scope+_name+'/emb_d' ) #conv1d
        self.emb_ld = tf.layers.Conv1D(dim,1,_scope=_scope+_name+'/emb_ld' ) #conv1d_2

        self.project_d = tf.layers.Conv1D(dim,1,_scope=_scope+_name+'/proj_d' ) #conv1d_1
        self.project_ld = tf.layers.Conv1D(dim,1,_scope=_scope+_name+'/proj_ld' ) #conv1d_3
        self.project_query = tf.layers.Dense(dim,_scope=_scope+_name+'/proj_q' ) #
        self.project_ref = tf.layers.Conv1D(dim,1,_scope=_scope+_name+'/proj_ref' ) #conv1d_4


        self.C = C  # tanh exploration parameter
        self.tanh = tf.nn.tanh

    def __call__(self, query, ref, env):
        """
        This function gets a query tensor and ref rensor and returns the logit op.
        Args: 
            query: is the hidden state of the decoder at the current
                time step. [batch_size x dim]
            ref: the set of hidden states from the encoder. 
                [batch_size x max_time x dim]

        Returns:
            e: convolved ref with shape [batch_size x max_time x dim]
            logits: [batch_size x max_time]
        """
        # get the current demand and load values from environment
        demand = env.demand
        load = env.load
        max_time = tf.shape(demand)[1]

        # embed demand and project it
        # emb_d:[batch_size x max_time x dim ]
        emb_d = self.emb_d(tf.expand_dims(demand,2))
        # d:[batch_size x max_time x dim ]
        d = self.project_d(emb_d)

        # embed load - demand
        # emb_ld:[batch_size*beam_width x max_time x hidden_dim]
        emb_ld = self.emb_ld(tf.expand_dims(tf.tile(tf.expand_dims(load,1),[1,max_time])-
                                              demand,2))
        # ld:[batch_size*beam_width x hidden_dim x max_time ] 
        ld = self.project_ld(emb_ld)

        # expanded_q,e: [batch_size x max_time x dim]
        e = self.project_ref(ref)
        q = self.project_query(query) #[batch_size x dim]
        expanded_q = tf.tile(tf.expand_dims(q,1),[1,max_time,1])

        # v_view:[batch_size x dim x 1]
        v_view = tf.tile( self.v, [tf.shape(e)[0],1,1]) 
        
        # u : [batch_size x max_time x dim] * [batch_size x dim x 1] = 
        #       [batch_size x max_time]
        u = tf.squeeze(tf.matmul(self.tanh(expanded_q + e + d + ld), v_view),2)

        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  

        return e, logits


class AttentionVRPCritic(object):
    """A generic attention module for the attention in vrp model"""
    def __init__(self, dim, use_tanh=False, C=10,_name='Attention',_scope=''):

        self.use_tanh = use_tanh
        self._scope = _scope

        with tf.variable_scope(_scope+_name):
            # self.v: is a variable with shape [1 x dim]
            self.v = tf.get_variable('v',[1,dim],
                       initializer=tf.contrib.layers.xavier_initializer())
            self.v = tf.expand_dims(self.v,2)
            
        self.emb_d = tf.layers.Conv1D(dim,1,_scope=_scope+_name +'/emb_d') #conv1d
        self.project_d = tf.layers.Conv1D(dim,1,_scope=_scope+_name +'/proj_d') #conv1d_1
        
        self.project_query = tf.layers.Dense(dim,_scope=_scope+_name +'/proj_q') #
        self.project_ref = tf.layers.Conv1D(dim,1,_scope=_scope+_name +'/proj_e') #conv1d_2

        self.C = C  # tanh exploration parameter
        self.tanh = tf.nn.tanh
        
    def __call__(self, query, ref, env):
        """
        This function gets a query tensor and ref rensor and returns the logit op.
        Args: 
            query: is the hidden state of the decoder at the current
                time step. [batch_size x dim]
            ref: the set of hidden states from the encoder. 
                [batch_size x max_time x dim]

            env: keeps demand ond load values and help decoding. Also it includes mask.
                env.mask: a matrix used for masking the logits and glimpses. It is with shape
                         [batch_size x max_time]. Zeros in this matrix means not-masked nodes. Any 
                         positive number in this mask means that the node cannot be selected as next 
                         decision point.
                env.demands: a list of demands which changes over time.

        Returns:
            e: convolved ref with shape [batch_size x max_time x dim]
            logits: [batch_size x max_time]
        """
        # we need the first demand value for the critic
        demand = env.input_data[:,:,-1]
        max_time = tf.shape(demand)[1]

        # embed demand and project it
        # emb_d:[batch_size x max_time x dim ]
        emb_d = self.emb_d(tf.expand_dims(demand,2))
        # d:[batch_size x max_time x dim ]
        d = self.project_d(emb_d)


        # expanded_q,e: [batch_size x max_time x dim]
        e = self.project_ref(ref)
        q = self.project_query(query) #[batch_size x dim]
        expanded_q = tf.tile(tf.expand_dims(q,1),[1,max_time,1])

        # v_view:[batch_size x dim x 1]
        v_view = tf.tile( self.v, [tf.shape(e)[0],1,1]) 
        
        # u : [batch_size x max_time x dim] * [batch_size x dim x 1] = 
        #       [batch_size x max_time]
        u = tf.squeeze(tf.matmul(self.tanh(expanded_q + e + d), v_view),2)

        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  

        return e, logits


###############################################################################
class Attention(object):
    """A generic attention module for a decoder in seq2seq models"""
    def __init__(self, dim, use_tanh=False, C=10,_name='Attention',_scope=''):
        self.use_tanh = use_tanh
        self._scope = _scope

        with tf.variable_scope(_scope+_name):
            # self.v: is a variable with shape [1 x dim]
            self.v = tf.get_variable('v',[1,dim],
                       initializer=tf.contrib.layers.xavier_initializer())
            self.v = tf.expand_dims(self.v,2)
        self.project_query = tf.layers.Dense(dim,_scope=_scope+_name +'/dense')
        self.project_ref = tf.layers.Conv1D(dim,1,_scope=_scope+_name +'/conv1d')
        self.C = C  # tanh exploration parameter
        self.tanh = tf.nn.tanh

    def __call__(self, query, ref, *args, **kwargs):
        """
        This function gets a query tensor and ref rensor and returns the logit op.
        Args: 
            query: is the hidden state of the decoder at the current
                time step. [batch_size x dim]
            ref: the set of hidden states from the encoder. 
                [batch_size x max_time x dim]

        Returns:
            e: convolved ref with shape [batch_size x max_time x dim]
            logits: [batch_size x max_time]
        """
        # expanded_q,e: [batch_size x max_time x dim]
        e = self.project_ref(ref)
        q = self.project_query(query) #[batch_size x dim]
        expanded_q = tf.tile(tf.expand_dims(q,1),[1,tf.shape(e)[1],1])

        # v_view:[batch_size x dim x 1]
        v_view = tf.tile( self.v, [tf.shape(e)[0],1,1]) 
        
        # u : [batch_size x max_time x dim] * [batch_size x dim x 1] = 
        #       [batch_size x max_time]
        u = tf.squeeze(tf.matmul(self.tanh(expanded_q + e), v_view),2)

        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  

        return e, logits

###############################################################################
class Embedding(object):
    '''
    This class is the base class for embedding the input graph.
    '''
    def __init__(self,emb_type, embedding_dim):
        self.emb_type = emb_type
        self.embedding_dim = embedding_dim

    def __call__(self,input_pnt):
        # returns the embeded tensor. Should be implemented in child classes
        pass

class LinearEmbedding(Embedding):
    '''
    This class implements linear embedding. It is only a mapping 
    to a higher dimensional space.
    '''
    def __init__(self,embedding_dim,_scope=''):
        '''
        Input: 
            embedding_dim: embedding dimension
        '''

        super(LinearEmbedding,self).__init__('linear',embedding_dim)
        self.project_emb = tf.layers.Conv1D(embedding_dim,1,
            _scope=_scope+'Embedding/conv1d')

    def __call__(self,input_pnt):
        # emb_inp_pnt: [batch_size, max_time, embedding_dim]
        emb_inp_pnt = self.project_emb(input_pnt)
        # emb_inp_pnt = tf.Print(emb_inp_pnt,[emb_inp_pnt])
        return emb_inp_pnt

###############################################################################
class RLAgent(object):
    
    def __init__(self,
                args,
                prt,
                env,
                dataGen,
                reward_func,
                clAttentionActor,
                clAttentionCritic,
                is_train=True,
                _scope=''):
        '''
        This class builds the model and run testt and train.
        Inputs:
            args: arguments. See the description in config.py file.
            prt: print controller which writes logs to a file.
            env: an instance of the environment.
            dataGen: a data generator which generates data for test and training.
            reward_func: the function which is used for computing the reward. In the 
                        case of TSP and VRP, it returns the tour length.
            clAttentionActor: Attention mechanism that is used in actor.
            clAttentionCritic: Attention mechanism that is used in critic.
            is_train: if true, the agent is used for training; else, it is used only 
                        for inference.
        '''
        
        self.args = args
        self.prt = prt
        self.env = env
        self.dataGen = dataGen
        self.reward_func = reward_func
        self.clAttentionCritic = clAttentionCritic
        
        self.embedding = LinearEmbedding(args['embedding_dim'],
            _scope=_scope+'Actor/')
        self.decodeStep = RNNDecodeStep(clAttentionActor,
                        args['hidden_dim'], 
                        use_tanh=args['use_tanh'],
                        tanh_exploration=args['tanh_exploration'],
                        n_glimpses=args['n_glimpses'],
                        mask_glimpses=args['mask_glimpses'], 
                        mask_pointer=args['mask_pointer'], 
                        forget_bias=args['forget_bias'], 
                        rnn_layers=args['rnn_layers'],
                        _scope='Actor/')
        self.decoder_input = tf.get_variable('decoder_input', [1,1,args['embedding_dim']],
                       initializer=tf.contrib.layers.xavier_initializer())

        start_time  = time.time()
        if is_train:
            self.train_summary = self.build_model(decode_type = "stochastic" )
            self.train_step = self.build_train_step()

        self.val_summary_greedy = self.build_model(decode_type = "greedy" )
        self.val_summary_beam = self.build_model(decode_type = "beam_search")

        model_time = time.time()- start_time
        self.prt.print_out("It took {}s to build the agent.".format(str(model_time)))

        self.saver = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            
        
    def build_model(self, decode_type = "greedy"):
        
        # builds the model
        args = self.args
        env = self.env
        batch_size = tf.shape(env.input_pnt)[0]

        # input_pnt: [batch_size x max_time x 2]
        input_pnt = env.input_pnt
        # encoder_emb_inp: [batch_size, max_time, embedding_dim]
        encoder_emb_inp = self.embedding(input_pnt)

        if decode_type == 'greedy' or decode_type == 'stochastic':
            beam_width = 1
        elif decode_type == 'beam_search': 
            beam_width = args['beam_width']
            
        # reset the env. The environment is modified to handle beam_search decoding.
        env.reset(beam_width)

        BatchSequence = tf.expand_dims(tf.cast(tf.range(batch_size*beam_width), tf.int64), 1)


        # create tensors and lists
        actions_tmp = []
        logprobs = []
        probs = []
        idxs = []

        # start from depot
        idx = (env.n_nodes-1)*tf.ones([batch_size*beam_width,1])
        action = tf.tile(input_pnt[:,env.n_nodes-1],[beam_width,1])


        # decoder_state
        initial_state = tf.zeros([args['rnn_layers'], 2, batch_size*beam_width, args['hidden_dim']])
        l = tf.unstack(initial_state, axis=0)
        decoder_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0],l[idx][1])
                  for idx in range(args['rnn_layers'])])            

        # start from depot in VRP and from a trainable nodes in TSP
        # decoder_input: [batch_size*beam_width x 1 x hidden_dim]
        if args['task_name'] == 'tsp':
            # decoder_input: [batch_size*beam_width x 1 x hidden_dim]
            decoder_input = tf.tile(self.decoder_input, [batch_size* beam_width,1,1])
        elif args['task_name'] == 'vrp':
            decoder_input = tf.tile(tf.expand_dims(encoder_emb_inp[:,env.n_nodes-1], 1), 
                                    [beam_width,1,1])

        # decoding loop
        context = tf.tile(encoder_emb_inp,[beam_width,1,1])
        for i in range(args['decode_len']):
            
            logit, prob, logprob, decoder_state = self.decodeStep.step(decoder_input,
                                context,
                                env,
                                decoder_state)
            # idx: [batch_size*beam_width x 1]
            beam_parent = None
            if decode_type == 'greedy':
                idx = tf.expand_dims(tf.argmax(prob, 1),1)
            elif decode_type == 'stochastic':
                # select stochastic actions. idx has shape [batch_size x 1]
                # tf.multinomial sometimes gives numerical errors, so we use our multinomial :(
                def my_multinomial():
                    prob_idx = tf.stop_gradient(prob)
                    prob_idx_cum = tf.cumsum(prob_idx,1)
                    rand_uni = tf.tile(tf.random_uniform([batch_size,1]),[1,env.n_nodes])
                    # sorted_ind : [[0,1,2,3..],[0,1,2,3..] , ]
                    sorted_ind = tf.cast(tf.tile(tf.expand_dims(tf.range(env.n_nodes),0),[batch_size,1]),tf.int64)
                    tmp = tf.multiply(tf.cast(tf.greater(prob_idx_cum,rand_uni),tf.int64), sorted_ind)+\
                        10000*tf.cast(tf.greater_equal(rand_uni,prob_idx_cum),tf.int64)

                    idx = tf.expand_dims(tf.argmin(tmp,1),1)
                    return tmp, idx

                tmp, idx = my_multinomial()
                # check validity of tmp -> True or False -- True mean take a new sample
                tmp_check = tf.cast(tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(tmp,1),(10000*env.n_nodes)-1),
                                                          tf.int32)),tf.bool)
                tmp , idx = tf.cond(tmp_check,my_multinomial,lambda:(tmp,idx))

            elif decode_type == 'beam_search':
                if i==0:
                    # BatchBeamSeq: [batch_size*beam_width x 1]
                    # [0,1,2,3,...,127,0,1,...],
                    batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(batch_size), tf.int64),
                                                         [beam_width]),1)
                    beam_path  = []
                    log_beam_probs = []
                    # in the initial decoder step, we want to choose beam_width different branches
                    # log_beam_prob: [batch_size, sourceL]
                    log_beam_prob = tf.log(tf.split(prob,num_or_size_splits=beam_width, axis=0)[0])

                elif i > 0:
                    log_beam_prob = tf.log(prob) + log_beam_probs[-1]
                    # log_beam_prob:[batch_size, beam_width*sourceL]
                    log_beam_prob = tf.concat(tf.split(log_beam_prob, num_or_size_splits=beam_width, axis=0),1)

                # topk_prob_val,topk_logprob_ind: [batch_size, beam_width]
                topk_logprob_val, topk_logprob_ind = tf.nn.top_k(log_beam_prob, beam_width)

                # topk_logprob_val , topk_logprob_ind: [batch_size*beam_width x 1]
                topk_logprob_val = tf.transpose(tf.reshape(
                    tf.transpose(topk_logprob_val), [1,-1]))

                topk_logprob_ind = tf.transpose(tf.reshape(
                    tf.transpose(topk_logprob_ind), [1,-1]))

                #idx,beam_parent: [batch_size*beam_width x 1]                               
                idx = tf.cast(topk_logprob_ind % env.n_nodes, tf.int64) # Which city in route.
                beam_parent = tf.cast(topk_logprob_ind // env.n_nodes, tf.int64) # Which hypothesis it came from.

                # batchedBeamIdx:[batch_size*beam_width]
                batchedBeamIdx= batchBeamSeq + tf.cast(batch_size,tf.int64)*beam_parent
                prob = tf.gather_nd(prob,batchedBeamIdx)

                beam_path.append(beam_parent)
                log_beam_probs.append(topk_logprob_val)

            state = env.step(idx,beam_parent)
            batched_idx = tf.concat([BatchSequence,idx],1)


            decoder_input = tf.expand_dims(tf.gather_nd(
                tf.tile(encoder_emb_inp,[beam_width,1,1]), batched_idx),1)

            logprob = tf.log(tf.gather_nd(prob, batched_idx))
            probs.append(prob)
            idxs.append(idx)
            logprobs.append(logprob)           

            action = tf.gather_nd(tf.tile(input_pnt, [beam_width,1,1]), batched_idx )
            actions_tmp.append(action)

        if decode_type=='beam_search':
            # find paths of the beam search
            tmplst = []
            tmpind = [BatchSequence]
            for k in reversed(range(len(actions_tmp))):

                tmplst = [tf.gather_nd(actions_tmp[k],tmpind[-1])] + tmplst
                tmpind += [tf.gather_nd(
                    (batchBeamSeq + tf.cast(batch_size,tf.int64)*beam_path[k]),tmpind[-1])]
            actions = tmplst
        else: 
            actions = actions_tmp

        R = self.reward_func(actions)            

        ### critic
        v = tf.constant(0)
        if decode_type=='stochastic':
            with tf.variable_scope("Critic"):
                with tf.variable_scope("Encoder"):
                    # init states
                    initial_state = tf.zeros([args['rnn_layers'], 2, batch_size, args['hidden_dim']])
                    l = tf.unstack(initial_state, axis=0)
                    rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0],l[idx][1])
                              for idx in range(args['rnn_layers'])])

                    hy = rnn_tuple_state[0][1]

                with tf.variable_scope("Process"):
                    for i in range(args['n_process_blocks']):

                        process = self.clAttentionCritic(args['hidden_dim'],_name="P"+str(i))
                        e,logit = process(hy, encoder_emb_inp, env)

                        prob = tf.nn.softmax(logit)
                        # hy : [batch_size x 1 x sourceL] * [batch_size  x sourceL x hidden_dim]  ->
                        #[batch_size x h_dim ]
                        hy = tf.squeeze(tf.matmul(tf.expand_dims(prob,1), e ) ,1)

                with tf.variable_scope("Linear"):
                    v = tf.squeeze(tf.layers.dense(tf.layers.dense(hy,args['hidden_dim']\
                                                               ,tf.nn.relu,name='L1'),1,name='L2'),1)


        return (R, v, logprobs, actions, idxs, env.input_pnt , probs)
    
    def build_train_step(self):
        '''
        This function returns a train_step op, in which by running it we proceed one training step.
        '''
        args = self.args
        
        R, v, logprobs, actions, idxs , batch , probs= self.train_summary

        v_nograd = tf.stop_gradient(v)
        R = tf.stop_gradient(R)

        # losses
        actor_loss = tf.reduce_mean(tf.multiply((R-v_nograd),tf.add_n(logprobs)),0)
        critic_loss = tf.losses.mean_squared_error(R,v)

        # optimizers
        actor_optim = tf.train.AdamOptimizer(args['actor_net_lr'])
        critic_optim = tf.train.AdamOptimizer(args['critic_net_lr'])

        # compute gradients
        actor_gra_and_var = actor_optim.compute_gradients(actor_loss,\
                                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor'))
        critic_gra_and_var = critic_optim.compute_gradients(critic_loss,\
                                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic'))

        # clip gradients
        clip_actor_gra_and_var = [(tf.clip_by_norm(grad, args['max_grad_norm']), var) \
                                  for grad, var in actor_gra_and_var]

        clip_critic_gra_and_var = [(tf.clip_by_norm(grad, args['max_grad_norm']), var) \
                                  for grad, var in critic_gra_and_var]

        # apply gradients
        actor_train_step = actor_optim.apply_gradients(clip_actor_gra_and_var)
        critic_train_step = critic_optim.apply_gradients(clip_critic_gra_and_var)

        train_step = [actor_train_step, 
                          critic_train_step ,
                          actor_loss, 
                          critic_loss,
                          actor_gra_and_var,
                          critic_gra_and_var,
                          R, 
                          v, 
                          logprobs,
                          probs,
                          actions,
                          idxs]
        return train_step

    def Initialize(self,sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.load_model()

    def load_model(self):
        latest_ckpt = tf.train.latest_checkpoint(self.args['load_path'])
        if latest_ckpt is not None:
            self.saver.restore(self.sess, latest_ckpt)
            
    def evaluate_single(self,eval_type='greedy'):
        start_time = time.time()
        avg_reward = []

        if eval_type == 'greedy':
            summary = self.val_summary_greedy
        elif eval_type == 'beam_search':
            summary = self.val_summary_beam
        self.dataGen.reset()
        #Reward_list = []
        for step in range(self.dataGen.n_problems):

            data = self.dataGen.get_test_next()
            R, v, logprobs, actions,idxs, batch, _= self.sess.run(summary,
                                         feed_dict={self.env.input_data:data,
                                                   self.decodeStep.dropout:0.0})
            if eval_type=='greedy':
                avg_reward.append(R)
                R_ind0 = 0
            elif eval_type=='beam_search':
                # R : [batch_size x beam_width]
                R = np.concatenate(np.split(np.expand_dims(R,1) ,self.args['beam_width'], axis=0),1 )
                R_val = np.amin(R,1, keepdims = False)
                R_ind0 = np.argmin(R,1)[0]
                avg_reward.append(R_val)
            
            #Reward_list.append(R[0])
            
            


            # sample decode
            if step % int(self.args['log_interval']) == 0:
                example_output = []
                example_input = []
                for i in range(self.env.n_nodes):
                    example_input.append(list(batch[0, i, :]))
                for idx, action in enumerate(actions):
                    example_output.append(list(action[R_ind0*np.shape(batch)[0]]))
                self.prt.print_out('\n\nVal-Step of {}: {}'.format(eval_type,step))
                self.prt.print_out('\nExample test input: {}'.format(example_input))
                self.prt.print_out('\nExample test output: {}'.format(example_output))
                self.prt.print_out('\nExample test reward: {} - best: {}'.format(R[0],R_ind0))

        end_time = time.time() - start_time
        
        df_reward = pd.DataFrame(avg_reward,columns=['reward'])
        df_reward.to_csv('reward.csv')

        # Finished going through the iterator dataset.
        self.prt.print_out('\nValidation overall avg_reward: {}'.format(np.mean(avg_reward)) )
        self.prt.print_out('Validation overall reward std: {}'.format(np.sqrt(np.var(avg_reward))) )

        self.prt.print_out("Finished evaluation with %d steps in %s." % (step\
                           ,time.strftime("%H:%M:%S", time.gmtime(end_time))))

        
    def evaluate_batch(self,eval_type='greedy'):
        self.env.reset()
        if eval_type == 'greedy':
            summary = self.val_summary_greedy
            beam_width = 1
        elif eval_type == 'beam_search':
            summary = self.val_summary_beam
            beam_width = self.args['beam_width']
            
            
        data = self.dataGen.get_test_all()
        start_time = time.time()
        R, v, logprobs, actions,idxs, batch, _= self.sess.run(summary,
                                     feed_dict={self.env.input_data:data,
                                               self.decodeStep.dropout:0.0})
        R = np.concatenate(np.split(np.expand_dims(R,1) ,beam_width, axis=0),1 )
        R = np.amin(R,1, keepdims = False)

        end_time = time.time() - start_time
        self.prt.print_out('Average of {} in batch-mode: {} -- std {} -- time {} s'.format(eval_type,\
            np.mean(R),np.sqrt(np.var(R)),end_time))        
        
    def inference(self, infer_type='batch'):
        if infer_type == 'batch':
            self.evaluate_batch('greedy')
            self.evaluate_batch('beam_search')
        elif infer_type == 'single':
            self.evaluate_single('greedy')
            self.evaluate_single('beam_search')
        self.prt.print_out("##################################################################")

    def run_train_step(self):
        data = self.dataGen.get_train_next()

        train_results = self.sess.run(self.train_step,
                                 feed_dict={self.env.input_data:data,
                                  self.decodeStep.dropout:self.args['dropout']})
        return train_results

###############################################################################
class DecodeStep(object):
    '''
    Base class of the decoding (without RNN)
    '''
    def __init__(self, 
            ClAttention,
            hidden_dim,
            use_tanh=False,
            tanh_exploration=10.,
            n_glimpses=0,
            mask_glimpses=True,
            mask_pointer=True,
            _scope=''):
        '''
        This class does one-step of decoding.
        Inputs:
            ClAttention:    the class which is used for attention
            hidden_dim:     hidden dimension of RNN
            use_tanh:       whether to use tanh exploration or not
            tanh_exploration: parameter for tanh exploration
            n_glimpses:     number of glimpses
            mask_glimpses:  whether to use masking for the glimpses or not
            mask_pointer:   whether to use masking for the glimpses or not
            _scope:         variable scope
        '''

        self.hidden_dim = hidden_dim
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_pointer = mask_pointer
        self._scope = _scope
        self.BIGNUMBER = 100000.


        # create glimpse and attention instances as well as tf.variables.
        ## create a list of class instances
        self.glimpses = [None for _ in range(self.n_glimpses)]
        for i in range(self.n_glimpses):
            self.glimpses[i] = ClAttention(hidden_dim, 
                use_tanh=False,
                _scope=self._scope,
                _name="Glimpse"+str(i))
            
        # build TF variables required for pointer
        self.pointer = ClAttention(hidden_dim, 
            use_tanh=use_tanh, 
            C=tanh_exploration,
            _scope=self._scope,
            _name="Decoder/Attention")

    def get_logit_op(self,
                     decoder_inp,
                     context,
                     Env,
                    *args,
                    **kwargs):
        """
        For a given input to deocoder, returns the logit op.
        Input:
            decoder_inp: it is the input problem with dimensions [batch_size x dim].
                        Usually, it is the embedded problem with dim = embedding_dim.
            context: the context vetor from the encoder. It is usually the output of rnn with
                      shape [batch_size x max_time x dim]
            Env: an instance of the environment. It should have:
                Env.mask: a matrix used for masking the logits and glimpses. It is with shape
                         [batch_size x max_time]. Zeros in this matrix means not-masked nodes. Any 
                         positive number in this mask means that the node cannot be selected as 
                         the next decision point.
        Returns:
            logit: the logits which will used by decoder for producing a solution. It has shape
            [batch_size x max_time].
        """

        # glimpses
        for i in range(self.n_glimpses):
            # ref: [batch_size x max_time x hidden_dim], logit : [batch_size x max_time]
            ref, logit = self.glimpses[i](decoder_inp, context,Env)
            if self.mask_glimpses:
                logit -= self.BIGNUMBER* Env.mask
            # prob: [batch_size x max_time
            prob = tf.nn.softmax(logit)
            # decoder_inp : [batch_size x 1 x max_time ] * [batch_size x max_time x hidden_dim] -> 
            #[batch_size x hidden_dim ]
            decoder_inp = tf.squeeze(tf.matmul( tf.expand_dims(prob,1),ref) ,1)

        # attention
        _, logit = self.pointer(decoder_inp,context,Env)
        if self.mask_pointer:
            logit -= self.BIGNUMBER* Env.mask

        return logit , None

    def step(self,
            decoder_inp,
            context,
            Env,
            decoder_state=None,
            *args,
            **kwargs):
        '''
        get logits and probs at a given decoding step.
        Inputs:
            decoder_input: Input of the decoding step with shape [batch_size x embedding_dim]
            context: context vector to use in attention
            Env: an instance of the environment
            decoder_state: The state of the LSTM cell. It can be None when we use a decoder without 
                LSTM cell.
        Returns:
            logit: logits with shape [batch_size x max_time]
            prob: probabilities for the next location visit with shape of [batch_size x max_time]
            logprob: log of probabilities
            decoder_state: updated state of the LSTM cell
        '''

        logit, decoder_state = self.get_logit_op(
                     decoder_inp,
                     context,
                     Env, 
                     decoder_state)

        logprob = tf.nn.log_softmax(logit)
        prob = tf.exp(logprob)

        return logit, prob, logprob, decoder_state

class RNNDecodeStep(DecodeStep):
    '''
    Decodes the sequence. It keeps the decoding history in a RNN.
    '''
    def __init__(self, 
            ClAttention,
            hidden_dim,
            use_tanh=False,
            tanh_exploration=10.,
            n_glimpses=0,
            mask_glimpses=True,
            mask_pointer=True,
            forget_bias=1.0,
            rnn_layers=1,
            _scope=''):

        '''
        This class does one-step of decoding which uses RNN for storing the sequence info.
        Inputs:
            ClAttention:    the class which is used for attention
            hidden_dim:     hidden dimension of RNN
            use_tanh:       whether to use tanh exploration or not
            tanh_exploration: parameter for tanh exploration
            n_glimpses:     number of glimpses
            mask_glimpses:  whether to use masking for the glimpses or not
            mask_pointer:   whether to use masking for the glimpses or not
            forget_bias:    forget bias of LSTM
            rnn_layers:     number of LSTM layers
            _scope:         variable scope

        '''

        super(RNNDecodeStep,self).__init__(ClAttention,
                                        hidden_dim,
                                        use_tanh=use_tanh,
                                        tanh_exploration=tanh_exploration,
                                        n_glimpses=n_glimpses,
                                        mask_glimpses=mask_glimpses,
                                        mask_pointer=mask_pointer,
                                        _scope=_scope)
        self.forget_bias = forget_bias
        self.rnn_layers = rnn_layers     
#         self.dropout = tf.placeholder(tf.float32,name='decoder_rnn_dropout')

        # build a multilayer LSTM cell
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, 
            forget_bias=forget_bias)
        self.dropout = tf.placeholder(tf.float32,name='decoder_rnn_dropout') 
        single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - self.dropout))
        self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * rnn_layers)

    def get_logit_op(self,
                    decoder_inp,
                    context,
                    Env,
                    decoder_state,
                    *args,
                    **kwargs):
        """
        For a given input to decoder, returns the logit op and new decoder_state.
        Input:
            decoder_inp: it is the input problem with dimensions [batch_size x dim].
                        Usually, it is the embedded problem with dim = embedding_dim.
            context: the context vetor from the encoder. It is usually the output of rnn with
                      shape [batch_size x max_time x dim]
            Env: an instance of the environment. It should have:
                Env.mask: a matrix used for masking the logits and glimpses. It is with shape
                         [batch_size x max_time]. Zeros in this matrix means not-masked nodes. Any 
                         positive number in this mask means that the node cannot be selected as 
                         the next decision point.
            decoder_state: The state as a list of size rnn_layers, and each element is a
                    LSTMStateTuples with  x 2 tensors with dimension of [batch_size x hidden_dim].
                    The first one corresponds to c and the second one is h.
        Returns:
            logit: the logits which will used by decoder for producing a solution. It has shape
                    [batch_size x max_time].
            decoder_state: the update decoder state.
        """

        #decoder_inp = tf.reshape(decoder_inp,[-1,1,self.hidden_dim]) 
        
        decoder_inp = (tf.reduce_mean(context,axis=1,keepdims=True)+decoder_inp)/2.0
        
        #decoder_inp = (tf.expand_dims(context[:,-1,:],1)+decoder_inp)/2.0
        #print(tf.reduce_mean(context,1))
        #print(decoder_inp)
        #tf.expand_dims(context[:,-1,:],1))
        _ , decoder_state = tf.nn.dynamic_rnn(self.cell,
                                              decoder_inp,
                                              initial_state=decoder_state,
                                              scope=self._scope+'Decoder/LSTM/rnn')
        hy = decoder_state[-1].h

        # glimpses
        for i in range(self.n_glimpses):
            # ref: [batch_size x max_time x hidden_dim], logit : [batch_size x max_time]
            ref, logit = self.glimpses[i](hy,context,Env)
            if self.mask_glimpses:
                logit -= self.BIGNUMBER* Env.mask
            prob = tf.nn.softmax(logit)
            
            # hy : [batch_size x 1 x max_time ] * [batch_size x max_time x hidden_dim] -> 
            #[batch_size x hidden_dim ]
            hy = tf.squeeze(tf.matmul( tf.expand_dims(prob,1),ref) ,1)

        # attention
        _, logit = self.pointer(hy,context,Env)
        if self.mask_pointer:
            logit -= self.BIGNUMBER* Env.mask
    
        return logit , decoder_state

###############################################################################

class printOut(object):
    def __init__(self,f=None ,stdout_print=True):
        ''' 
        This class is used for controlling the printing. It will write in a 
        file f and screen simultanously.
        '''
        self.out_file = f
        self.stdout_print = stdout_print

    def print_out(self, s, new_line=True):
        """Similar to print but with support to flush and output to a file."""
        if isinstance(s, bytes):
            s = s.decode("utf-8")

        if self.out_file:
            self.out_file.write(s)
            if new_line:
                self.out_file.write("\n")
        self.out_file.flush()

        # stdout
        if self.stdout_print:
            print(s, end="", file=sys.stdout)
            if new_line:
                sys.stdout.write("\n")
            sys.stdout.flush()

    def print_time(self,s, start_time):
        """Take a start time, print elapsed duration, and return a new time."""
        self.print_out("%s, time %ds, %s." % (s, (time.time() - start_time) +"  " +str(time.ctime()) ))
        return time.time()


def get_time():
    '''returns formatted current time'''
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
 

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
        # GPU options:
        # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
        config_proto = tf.ConfigProto(
                        log_device_placement=log_device_placement,
                        allow_soft_placement=allow_soft_placement)
        config_proto.gpu_options.allow_growth = True
        return config_proto

def debug_tensor(s, msg=None, summarize=10):
        """Print the shape and value of a tensor at test time. Return a new tensor."""
        if not msg:
                msg = s.name
        return tf.Print(s, [tf.shape(s), s], msg + " ", summarize=summarize)

def has_nan(datum, tensor):
        if hasattr(tensor, 'dtype'):
                if (np.issubdtype(tensor.dtype, np.float) or
                        np.issubdtype(tensor.dtype, np.complex) or
                        np.issubdtype(tensor.dtype, np.integer)):
                        return np.any(np.isnan(tensor))
                else:
                        return False
        else:
                return False

def openAI_entropy(logits):
        # Entropy proposed by OpenAI in their A2C baseline
        a0 = logits - tf.reduce_max(logits, 2, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 2, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_mean(tf.reduce_sum(p0 * (tf.log(z0) - a0), 2))


def softmax_entropy(p0):
        # Normal information theory entropy by Shannon
        return - tf.reduce_sum(p0 * tf.log(p0 + 1e-6), axis=1)

def Dist_mat(A):
        # A is of shape [batch_size x nnodes x 2].
        # return: a distance matrix with shape [batch_size x nnodes x nnodes]
        nnodes = tf.shape(A)[1]
        A1 = tf.tile(tf.expand_dims(A,1),[1,nnodes,1,1])
        A2 = tf.tile(tf.expand_dims(A,2),[1,1,nnodes,1])
        dist = tf.norm(A1-A2,axis=3)
        return dist

###############################################################################
# task specific params
TaskVRP = namedtuple('TaskVRP', ['task_name', 
						'input_dim',
						'n_nodes' ,
						'n_cust',
						'decode_len',
						'capacity',
						'demand_max'])


task_lst = {}

# VRP10
vrp10 = TaskVRP(task_name = 'vrp',
			  input_dim=3,
			  n_nodes=11,
			  n_cust = 10,
			  decode_len=16,
			  capacity=20,
			  demand_max=9)
task_lst['vrp10'] = vrp10

# VRP20
vrp20 = TaskVRP(task_name = 'vrp',
			  input_dim=3,
			  n_nodes=21,
			  n_cust = 20,
			  decode_len=30,
			  capacity=30,
			  demand_max=9)
task_lst['vrp20'] = vrp20

# VRP50
vrp50 = TaskVRP(task_name = 'vrp',
			  input_dim=3,
			  n_nodes=51,
			  n_cust = 50,
			  decode_len=70,
			  capacity=40,
			  demand_max=9)
task_lst['vrp50'] = vrp50

# VRP100
vrp100 = TaskVRP(task_name = 'vrp',
			  input_dim=3,
			  n_nodes=101,
			  n_cust = 100,
			  decode_len=140,
			  capacity=50,
			  demand_max=9)
task_lst['vrp100'] = vrp100

###############################################################################

def str2bool(v):
    return v.lower() in ('true', '1')

def initialize_task_settings(args,task):

    try:
        task_params = task_lst[task]
    except:
        raise Exception('Task is not implemented.') 

    for name, value in task_params._asdict().items():
    	args[name] = value

    return args

def ParseParams():
    parser = argparse.ArgumentParser(description="Neural Combinatorial Optimization with RL")

    # Data
    parser.add_argument('--task', default='vrp10', help="Select the task to solve; i.e. vrp50")
    parser.add_argument('--batch_size', default=128,type=int, help='Batch size in training')
    parser.add_argument('--n_train', default=260000,type=int, help='Number of training steps')
    parser.add_argument('--test_size', default=1000,type=int, help='Number of problems in test set')

    # Network
    parser.add_argument('--agent_type', default='attention', help="attention|pointer")
    parser.add_argument('--forget_bias', default=1.0,type=float, help="Forget bias for BasicLSTMCell.")
    parser.add_argument('--embedding_dim', default=128,type=int, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', default=128,type=int, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_process_blocks', default=3,type=int,                     
                        help='Number of process block iters to run in the Critic network')
    parser.add_argument('--rnn_layers', default=1, type=int, help='Number of LSTM layers in the encoder and decoder')
    parser.add_argument('--decode_len', default=None,type=int,                     
                        help='Number of time steps the decoder runs before stopping')
    parser.add_argument('--n_glimpses', default=0, type=int, help='Number of glimpses to use in the attention')
    parser.add_argument('--tanh_exploration', default=10.,  type=float,                   
             help='Hyperparam controlling exploration in the net by scaling the tanh in the softmax')
    parser.add_argument('--use_tanh', type=str2bool, default=False, help='')
    parser.add_argument('--mask_glimpses', type=str2bool, default=True, help='')
    parser.add_argument('--mask_pointer', type=str2bool, default=True, help='')
    parser.add_argument('--dropout', default=0.1, type=float, help='The dropout prob')

    # Training
    parser.add_argument('--is_train', default=True,type=str2bool, help="whether to do the training or not")
    parser.add_argument('--actor_net_lr', default=1e-4,type=float, help="Set the learning rate for the actor network")
    parser.add_argument('--critic_net_lr', default=1e-4,type=float, help="Set the learning rate for the critic network")
    parser.add_argument('--random_seed', default=24601,type=int, help='')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, help='Gradient clipping')
    parser.add_argument('--entropy_coeff', default=0.0, type=float, help='coefficient for entropy regularization')
    # parser.add_argument('--loss_type', type=int, default=1, help='1,2,3')

    # inference
    parser.add_argument('--infer_type', default='batch', 
        help='single|batch: do inference for the problems one-by-one, or run it all at once')
    parser.add_argument('--beam_width', default=10, type=int, help='')

    # Misc
    parser.add_argument('--stdout_print', default=True, type=str2bool, help='print control')
    parser.add_argument("--gpu", default='3', type=str,help="gpu number.")
    parser.add_argument('--log_interval', default=200,type=int, help='Log info every log_step steps')
    parser.add_argument('--test_interval', default=200,type=int, help='test every test_interval steps')
    parser.add_argument('--save_interval', default=10000,type=int, help='save every save_interval steps')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--load_path', type=str, default='', help='Path to load trained variables')
    parser.add_argument('--disable_tqdm', default=True, type=str2bool)
                        
    args, unknown = parser.parse_known_args()
    args = vars(args)

    args['log_dir'] = "{}/{}-{}".format(args['log_dir'],args['task'], get_time())
    if args['model_dir'] =='':
        args['model_dir'] = os.path.join(args['log_dir'],'model')

    # file to write the stdout
    try:
        os.makedirs(args['log_dir'])
        os.makedirs(args['model_dir'])
    except:
        pass

    # create a print handler
    out_file = open(os.path.join(args['log_dir'], 'results.txt'),'w+') 
    prt = printOut(out_file,args['stdout_print'])

    os.environ["CUDA_VISIBLE_DEVICES"]=  args['gpu'] 

    args = initialize_task_settings(args,args['task'])

    # print the run args
    for key, value in sorted(args.items()):
        prt.print_out("{}: {}".format(key,value))

    return args, prt

###############################################################################

def load_task_specific_components(task):
    if task == 'vrp':
        AttentionActor = AttentionVRPActor
        AttentionCritic = AttentionVRPCritic

    else:
        raise Exception('Task is not implemented')


    return DataGenerator, Env, reward_func, AttentionActor, AttentionCritic

def main(args, prt):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # load task specific classes
    DataGenerator, Env, reward_func, AttentionActor, AttentionCritic = \
        load_task_specific_components(args['task_name'])

    dataGen = DataGenerator(args)
    dataGen.reset()
    env = Env(args)
    # create an RL agent
    agent = RLAgent(args,
                    prt,
                    env,
                    dataGen,
                    reward_func,
                    AttentionActor,
                    AttentionCritic,
                    is_train=args['is_train'])
    agent.Initialize(sess)

    # train or evaluate
    start_time = time.time()
    if args['is_train']:
        prt.print_out('Training started ...')
        train_time_beg = time.time()
        for step in range(args['n_train']):
            summary = agent.run_train_step()
            _, _ , actor_loss_val, critic_loss_val, actor_gra_and_var_val, critic_gra_and_var_val,\
                R_val, v_val, logprobs_val,probs_val, actions_val, idxs_val= summary

            if step%args['save_interval'] == 0:
                agent.saver.save(sess,args['model_dir']+'/model.ckpt', global_step=step)

            if step%args['log_interval'] == 0:
                train_time_end = time.time()-train_time_beg
                prt.print_out('Train Step: {} -- Time: {} -- Train reward: {} -- Value: {}'\
                      .format(step,time.strftime("%H:%M:%S", time.gmtime(\
                        train_time_end)),np.mean(R_val),np.mean(v_val)))
                prt.print_out('    actor loss: {} -- critic loss: {}'\
                      .format(np.mean(actor_loss_val),np.mean(critic_loss_val)))
                train_time_beg = time.time()
            if step%args['test_interval'] == 0:
                agent.inference(args['infer_type'])

    else: # inference
        prt.print_out('Evaluation started ...')
        agent.inference(args['infer_type'])


    prt.print_out('Total time is {}'.format(\
        time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))

if __name__ == "__main__":
    args, prt = ParseParams()
    # Random
    random_seed = args['random_seed']
    if random_seed is not None and random_seed > 0:
        prt.print_out("# Set random seed to %d" % random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
    tf.reset_default_graph()

    main(args, prt)