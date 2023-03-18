from main import Task
from argparse import ArgumentParser
from module.dataset import load_audio
import torch
import numpy as np

def cosine_similarity(vector_1, vector_2):
    vector_1 = vector_1.detach().numpy()
    vector_2 = vector_2.detach().numpy()
    score = vector_1.dot(vector_2.T)
    denom = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    print(score)
    print(denom)
    score = score/denom
    return score

parser = ArgumentParser(add_help=True)
parser.add_argument('--embedding_dim', type=int, default=192)
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--second', type=int, default=3)
parser.add_argument("--loss_name", type=str, default="amsoftmax")
parser.add_argument("--input_layer", type=str, default="conv2d2")
parser.add_argument("--pos_enc_layer_type", type=str, default="abs_pos")
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--audio_path_1", type=str, default=None)
parser.add_argument("--audio_path_2", type=str, default=None)

hparams = parser.parse_args()

lightning_model = Task(
                    embedding_dim=hparams.embedding_dim,
                    num_blocks=hparams.num_blocks,
                    loss_name=hparams.loss_name,
                    input_layer=hparams.input_layer,
                    pos_enc_layer_type=hparams.pos_enc_layer_type
                )
lightning_model.eval()
hparams.checkpoint_path = "experiment/fbank_conv2d2/conformer_cat_6_192_amsoftmax/epoch=16_cosine_eer=1.26.ckpt"
state_dict = torch.load(hparams.checkpoint_path, map_location="cpu")["state_dict"]
lightning_model.load_state_dict(state_dict, strict=True)
print("load weight from {}".format(hparams.checkpoint_path))

audio_path_1 =  'datasets/test_set_voxceleb1/id10270/5r0dWxy17C8/00001.wav' # hparams['audio_path_1']
audio_path_2 =  'datasets/test_set_voxceleb1/id10271/1gtz-CUIygI/00001.wav' # hparams['audio_path_2']

wav_1 = load_audio(audio_path_1, hparams.second)
wav_2 = load_audio(audio_path_2, hparams.second)

emb_wav_1 = lightning_model(torch.FloatTensor(wav_1).unsqueeze(0))
emb_wav_2 = lightning_model(torch.FloatTensor(wav_2).unsqueeze(0))
sim = cosine_similarity(emb_wav_1, emb_wav_2)
print(sim)