from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union
import torch.distributed as dist
import random
from tqdm import tqdm 

import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn import functional as F

from torch.utils.data import DataLoader
from module.dataset import Evaluation_Dataset
import score as score
import time
import os

from main import Task

parser = ArgumentParser(add_help=True)
parser.add_argument('--embedding_dim', type=int, default=192)
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--second', type=int, default=-1)
parser.add_argument("--loss_name", type=str, default="amsoftmax")
parser.add_argument("--input_layer", type=str, default="conv2d2")
parser.add_argument("--pos_enc_layer_type", type=str, default="abs_pos")
parser.add_argument("--trial_path", type=str, help='path to your trial file')
parser.add_argument("--checkpoint_path", type=str, help='path to the pretrained model')
parser.add_argument("--sample_rate", type=int, help='audio input frequency (8000 Hz or 16000 Hz)')

hparams = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using ', device)
# Loading model
lightning_model = Task(
                    embedding_dim=hparams.embedding_dim,
                    num_blocks=hparams.num_blocks,
                    loss_name=hparams.loss_name,
                    input_layer=hparams.input_layer,
                    pos_enc_layer_type=hparams.pos_enc_layer_type,
                    trial_path=hparams.trial_path,
                    sample_rate=hparams.sample_rate
                )
state_dict = torch.load(hparams.checkpoint_path, map_location=device)["state_dict"]
lightning_model.load_state_dict(state_dict)
lightning_model.to(device)
lightning_model.eval()
print("load weight from {}".format(hparams.checkpoint_path))

trials = np.loadtxt(hparams.trial_path, str)
eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
print("number of enroll: {}".format(len(set(trials.T[1]))))
print("number of test: {}".format(len(set(trials.T[2]))))
print("number of evaluation: {}".format(len(eval_path)))
test_dataset = Evaluation_Dataset(eval_path, second=-1, sample_rate=hparams.sample_rate)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                        num_workers=10,
                                        shuffle=False, 
                                        batch_size=1)

test_vectors = []
index_mapping = {}
inference_times = []
audio_lens = []
audio_size = []
for batch_idx, batch in enumerate(tqdm(test_loader)):
    x, path = batch
    audio_lens.append(x.shape[1]/16000)
    path = path[0]
    start = time.time()
    with torch.no_grad():
        x = lightning_model(x.to(device))
    inference_times.append(time.time()-start)
    x = x.detach().cpu().numpy()[0]
    test_vectors.append(x)
    index_mapping[path] = batch_idx
    wav_size = os.stat(path).st_size * 0.001
    audio_size.append(wav_size)

print('Average inference time: ', sum(inference_times)/len(inference_times))
print('Max inference time: ', max(inference_times))
print('Min inference time: ', min(inference_times))

print('Average audio length: ', sum(audio_lens)/len(audio_lens))
print('Max audio length: ', max(audio_lens))
print('Min audio length: ', min(audio_lens))

print('Average audio size: ', sum(audio_size)/len(audio_size))
print('Max audio size: ', max(audio_size))
print('Min audio size: ', min(audio_size))

test_vectors = test_vectors - np.mean(test_vectors, axis=0)
labels, scores = score.cosine_score(
    trials, index_mapping, test_vectors)
EER, threshold = score.compute_eer(labels, scores)


print("\ncosine EER: {:.2f}% with threshold {:.2f}".format(EER*100, threshold))
print("cosine_eer", EER*100)
