# This file contains ShowAttendTell and AllImg model

# ShowAttendTell is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

# AllImg is a model where
# img feature is concatenated with word embedding at every time step as the input of lstm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import math


class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    def beam_search(self, state, logprobs, *args, **kwargs):
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt

        ## Calculates distance between 2 vectors
        def calc_distance(x, y):
            nx = np.asarray(x).reshape(1, -1)
            ny = np.asarray(y).reshape(1, -1)
            dist = euclidean_distances(nx, ny)
            return dist[0][0]

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    # compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob.cpu()
                    candidates.append(dict(c=ix[q, c], q=q,
                                           p=candidate_logprob,
                                           r=local_logprob))
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.clone() for _ in state]
            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        def cluster_beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, \
                              num_clusters, embeds, vocab, prev_beams):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #num_cluster: number of clusters
            #embeds: glove embeddings for vocab
            #vocab: set of vocab
            #prev_beams: previous beams in list[str] form
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size*2, ys.size(1))
            rows = beam_size*2
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    # compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob.cpu()
                    candidates.append(dict(c=ix[q, c], q=q,
                                           p=candidate_logprob,
                                           r=local_logprob))
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            print(candidates)

            print(list(embeds.keys())[:10])

            ## Original beam (for debugging)
            orig_beams = []
            for i in range(beam_size*2):
                orig_beam = vocab[candidates[i]['c'].item()]
                if t >= 1:
                    prev_beam = prev_beams[candidates[i]['q']]
                    orig_beams.append(prev_beam + [orig_beam])
                else:
                    orig_beams.append([orig_beam])
            print(orig_beams)

            ## Gets averaged beam embeddings
            beam_embeds = []
            for i in range(len(orig_beams)):
                be = [embeds[v] for v in orig_beams[i]]
                avg_be = []
                for k in range(len(be[0])):
                    avg_be.append(sum([be[j][k] for j in range(be)])/len(be))
                beam_embeds.append(avg_be)
            std_embeds = whited(beam_embeds)

            ## Cluster beam embeddings
            ## Run K-means to cluster candidates into K clusters
            centroids,_ = kmeans(std_embeds, self.num_clusters)
            cluster_labels = []
            for e in std_embeds:
                min_distance = 1e10
                label = -1
                for i, c in enumerate(centroids):
                    dist = self.calc_distance(c, e)

                    if dist < min_distance:
                        min_distance = dist
                        label = i
                cluster_labels.append(label)

            ## Get top BEAM_SIZE/NUM_CLUSTERS candidates from each cluster
            new_candidates = []
            cluster_counts = [0 for i in range(num_clusters)]
            indices = []
            for i, l in enumberate(cluster_labels):
                if cluster_counts[l] < math.ceil(beam_size / num_clusters):
                    new_candidates.append(candidates[i])
                    indices.append(i)
                    cluster_counts[l] += 1
                elif min(cluster_counts) == math.ceil(beam_size / num_clusters):
                    break
                else:
                    continue

            ## If there aren't enough in each cluster, add candidates with highest scores
            if len(new_candidates) < beam_size:
                while len(new_candidates) != beam_size:
                    for i, l in enumerate(cluster_labels):
                        if i not in indices:
                            new_candidates.append(candidates[i])
                            break
            
            candidates = new_candidates

            ## New beam
            new_beams = []
            for i in range(beam_size*2):
                new_beam = vocab[candidates[i]['c'].item()]
                if t >= 1:
                    prev_beam = prev_beams[candidates[i]['q']]
                    new_beam.append(prev_beam + [new_beam])
                else:
                    new_beams.append([new_beam])
            print(new_beams)
                
            
            new_state = [_.clone() for _ in state]
            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state

            print(beam_seq)
            print(beam_seq_logprobs)

            a = b
            
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates, cur_beams

        # start beam search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        num_clusters = opt.get('num_clusters', 1)

        if num_clusters > 1:
            print("Running cluster beam search!")
            vocab = opt.get('vocab')
            embeds = opt.get('embeds')
            

        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
        # running sum of logprobs for each beam
        beam_logprobs_sum = torch.zeros(beam_size)
        done_beams = []

        ## Keeps track of previous beams
        prev_beams = [""]

        for t in range(self.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            logprobsf = logprobs.data.float() # lets go to CPU for more efficiency in indexing operations
            # suppress UNK tokens in the decoding
            logprobsf[:,logprobsf.size(1)-1] =  logprobsf[:, logprobsf.size(1)-1] - 1000  

            ## Run special clustered beam step!
            if num_clusters > 1:
                beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, \
                          candidates_divm = cluster_beam_step(logprobsf, beam_size, \
                                                              t, beam_seq, beam_seq_logprobs, \
                                                              beam_logprobs_sum, state, \
                                                              num_clusters, embeds, vocab, prev_beams)
            else:
                beam_seq,\
                beam_seq_logprobs,\
                beam_logprobs_sum,\
                state,\
                candidates_divm = beam_step(logprobsf,
                                            beam_size,
                                            t,
                                            beam_seq,
                                            beam_seq_logprobs,
                                            beam_logprobs_sum,
                                            state)
                

            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(), 
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': float(beam_logprobs_sum[vix].cpu().numpy())
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # encode as vectors
            it = beam_seq[t]
            logprobs, state = self.get_logprobs_state(Variable(it.cuda()), *(args + (state,)))

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams
