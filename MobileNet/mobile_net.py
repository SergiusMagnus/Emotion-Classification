import librosa
import numpy as np
import torch
import torch.nn as nn
import torchvision

from PIL import Image
from pydub import AudioSegment

def preprocess_audio(path):
    _, sr = librosa.load(path)
    raw_audio = AudioSegment.from_file(path)
    
    samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
    trimmed, _ = librosa.effects.trim(samples, top_db=25)
    padded = np.pad(trimmed, (0, 180000-len(trimmed)), 'constant')
    return padded, sr

def get_spec(path):
    y, sr = preprocess_audio(path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    spec = ((S_dB - S_dB.min()) / (S_dB.max() - S_dB.min()) * 255).astype('uint8')
    img = Image.fromarray(spec).transpose(Image.FLIP_TOP_BOTTOM)
    return torchvision.transforms.functional.pil_to_tensor(img.convert('RGB')) / 255

def classify(path: str) -> str:
    model = torch.load('MobileNet\data\model_100_epoch_model.pth')
    logits = model((get_spec(path)).unsqueeze(0))
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)

    emotion_dic = {
        'neutral' : 0,
        'happy'   : 1,
        'sad'     : 2, 
        'angry'   : 3, 
        'fear'    : 4, 
        'disgust' : 5
    }

    key_list = list(emotion_dic.keys())
    val_list = list(emotion_dic.values())
 
    position = val_list.index(y_pred[0])

    return key_list[position]