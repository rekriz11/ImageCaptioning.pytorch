from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import unicodedata
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import json


def language_eval(dataset, preds, model_id, split):
    import sys
    if 'coco' in dataset:
        sys.path.append("coco-caption")
        annFile = 'coco-caption/annotations/captions_val2014.json'
    else:
        sys.path.append("f30k-caption")
        annFile = 'f30k-caption/annotations/dataset_flickr30k.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


def eval_split(model, crit, loader, eval_kwargs={}):
    print('Evaluation Arguments {}'.format(eval_kwargs))
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    output_json_file_path = eval_kwargs.get('output_json_file_path', 'output.json')

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []

    output_data = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
            with torch.no_grad():
                tmp = [Variable(torch.from_numpy(_)).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks = tmp

                loss = crit(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
        all_candidate_sentences_pre = None
        with torch.no_grad():
            tmp = [Variable(torch.from_numpy(_)).cuda() for _ in tmp]
            fc_feats, att_feats = tmp
            # forward the model to also get generated samples for each image
            result = model.sample(fc_feats, att_feats, eval_kwargs)
            if len(result) == 4:
                seq, _, all_candidate_sentences_pre,  all_candidate_scores_pre = result
            else:
                seq, _ = result
        seq = seq.cpu().numpy()
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        all_candidate_sentences = None
        all_candidate_scores = None
        if all_candidate_sentences_pre is not None:
            all_candidate_sentences = []
            all_candidate_scores = []
            for l in range(len(all_candidate_sentences_pre)):
                candidate_sentences = []
                candidate_scores = []
                for m in range(len(all_candidate_sentences_pre[l])):
                    candidate_sentences.append(utils.decode_sequence(loader.get_vocab(), all_candidate_sentences_pre[l][m].cpu().numpy()))
                    candidate_scores.append(all_candidate_scores_pre[l][m].cpu().numpy().item())
                all_candidate_sentences.append(candidate_sentences)
                all_candidate_scores.append(candidate_scores)

        for k, sent in enumerate(sents):
            image_output_data = {}
            image_output_data['input'] = os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path'])
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if all_candidate_sentences is not None:
                entry['captions'] = all_candidate_sentences[k]
                entry['scores'] = all_candidate_scores[k]
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
                                            data['infos'][k]['file_path']) + '" vis/imgs/img' + str(
                    len(predictions)) + '.jpg'  # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image {} best caption: {}'.format(entry['image_id'], entry['caption']))

            final_captions = []
            final_scores = []
            if 'captions' in entry:
                for o in range(len(entry['captions'])):
                    filtered_caption_list = [a for a in entry['captions'][o] if len(a) > 0]
                    if len(filtered_caption_list) == 1:
                        if verbose:
                            print('\t{} --> {}'.format(filtered_caption_list[0], entry['scores'][o]))
                        final_captions.append(filtered_caption_list[0].split())
                        final_scores.append(entry['scores'][o])
                    else:
                        if verbose:
                            print('!!!Something went wrong --> \t{}'.format(' '.join(filtered_caption_list)))
            image_output_data['pred'] = final_captions
            image_output_data['scores'] = final_scores
            output_data.append(image_output_data)

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if 0 <= num_images <= n:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    with open(output_json_file_path, 'w') as fout:
        print('Wrote to {}'.format(output_json_file_path))
        json.dump(output_data, fout)
    return loss_sum / loss_evals, predictions, lang_stats
