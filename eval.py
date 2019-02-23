#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch
import numpy
import random

## Loads in embeddings for all words in vocab
def load_embeddings(embeddings_file, vocab):
    print("Loading embeddings...")
    embeds = {}
    c = 0
    for i, line in enumerate(open(embeddings_file, 'rb')):
        splitLine = line.split()
        word = splitLine[0].decode('ascii', 'ignore')
        embedding = np.array([float(val) for val in splitLine[1:]])
        embeds[word] = embedding

    ## Filters by words in vocab, and initializes words that don't
    ## have glove embeddings
    vocab_embeds = {}
    found, only_lower, not_found = 0, 0, 0
    for k,v in vocab.items():
        try:
            vocab_embeds[v] = embeds[v]
            found += 1
        except KeyError:
            try:
                vocab_embeds[v] = embeds[v.lower()]
                only_lower += 1
            except KeyError:
                not_found += 1
                vocab_embeds[v] = np.array([0.0 for i in range(300)])

    return vocab_embeds

numpy.random.seed(37)
random.seed(37)

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, required=True,
                    help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                    help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, required=True,
                    help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=0,
                    help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                    help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=1,
                    help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                    help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                    help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                    help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=8,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--hidden_state_noise', type=float, default=0.0,
                help='Use this at decoding to add random noise to language model weights.')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='',
                    help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='',
                    help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='',
                    help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test',
                    help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='',
                    help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')

parser.add_argument('--output_json_file_path', type=str, default='output.json')
parser.add_argument('--number_of_samples', type=int, default=4, help='Number of samples when sample_max is 0 (for random sampling)')
parser.add_argument('--top_c', type=int, default=-1, help='Top c random sampling (when sample max is 0)')

# Options related to clustered beam search
parser.add_argument('--num_clusters', type=int, default=1,
                    help='Number of clusters if using clustered beam search')
parser.add_argument('--cluster_embeddings_file', type=str, default='',
                    help='Embeddings file if using clustered beam search.')

# Option related to k_per_candidate beam search
parser.add_argument('--k_per_cand', type=int, default=0,
                    help='Number of candidates allowed per beam hypothesis')

opt = parser.parse_args()

print("NUMBER OF CLUSTERS:")
print(opt.num_clusters)

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = cPickle.load(f)

# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id
assert not(opt.beam_size == 1 and opt.hidden_state_noise > 0.0)
 
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = dict()
for k,v in infos['vocab'].items():
    ind = int(k)
    word = str(v)
    vocab[ind] = word

opt.vocab = vocab # ix -> word mapping


opt.embeds = dict()
if opt.num_clusters > 1:
    ## Loads embeddings for all words in vocab
    embeds_file = opt.cluster_embeddings_file
    opt.embeds = load_embeddings(embeds_file, opt.vocab)

    '''
    embeds = dict()
    for k,v in vocab.items():
        embeds[v] = [0.0 for i in range(300)]
    opt.embeds = embeds
    '''

# Setup the model
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
    loader = DataLoader(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']


# Set sample options
loss, split_predictions, lang_stats = eval_utils.eval_split(
    model, crit, loader,
    vars(opt))

print('loss: ', loss)
if lang_stats:
    print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
