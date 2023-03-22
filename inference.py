from main import Task
from argparse import ArgumentParser
from module.dataset import load_audio
import torch
import numpy as np

def cosine_similarity(vector_1, vector_2):
    vector_1 = vector_1.detach().cpu().numpy()
    vector_2 = vector_2.detach().cpu().numpy()
    score = vector_1.dot(vector_2.T)
    denom = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    score = score/denom
    return score

parser = ArgumentParser(add_help=True)
parser.add_argument('--embedding_dim', type=int, default=192)
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--second', type=int, default=3)
parser.add_argument("--loss_name", type=str, default="amsoftmax")
parser.add_argument("--input_layer", type=str, default="conv2d2")
parser.add_argument("--pos_enc_layer_type", type=str, default="abs_pos")
parser.add_argument("--checkpoint_path", type=str, default='checkpoints/epoch=17_cosine_eer=1.24.ckpt')
parser.add_argument("--threshold", type=int, default=0.22)
parser.add_argument("--audio_path_1", type=str, default='audio_samples/spk1_utt1.wav')
parser.add_argument("--audio_path_2", type=str, default='audio_samples/spk1_utt2.wav')

hparams = parser.parse_args()

threshold = hparams.threshold
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Loading model
lightning_model = Task(
                    embedding_dim=hparams.embedding_dim,
                    num_blocks=hparams.num_blocks,
                    loss_name=hparams.loss_name,
                    input_layer=hparams.input_layer,
                    pos_enc_layer_type=hparams.pos_enc_layer_type
                )
lightning_model.eval()
state_dict = torch.load(hparams.checkpoint_path, map_location=device)["state_dict"]
lightning_model.load_state_dict(state_dict)
lightning_model.to(device)
print("load weight from {}".format(hparams.checkpoint_path))

# Running inference
audio_path_1 = hparams.audio_path_1
wav_1 = load_audio(audio_path_1, hparams.second)
emb_wav_1 = lightning_model(torch.FloatTensor(wav_1).unsqueeze(0).to(device))

audio_path_2 = hparams.audio_path_2
wav_2 = load_audio(audio_path_2, hparams.second)
emb_wav_2 = lightning_model(torch.FloatTensor(wav_2).unsqueeze(0).to(device))

sim = cosine_similarity(emb_wav_1, emb_wav_2)
print(True if sim >= threshold else False)