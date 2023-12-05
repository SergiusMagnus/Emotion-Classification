import librosa
import numpy as np
import torch
import torch.nn as nn
import torchvision

from PIL import Image
from pydub import AudioSegment

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True) # batch_first=True (batch_dim, seq_dim, feature_dim)
        
        self.init_linear = nn.Linear(input_dim, input_dim)
        
        # Readout layer
        self.linear = nn.Linear(self.hidden_dim * 2, output_dim)


    def forward(self, x):
        # Initialize hidden state with zeros
        linear_input = self.init_linear(x)

        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size ,hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (batch_size, num_layers, hidden_dim).
        lstm_out, self.hidden = self.lstm(linear_input)

        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[:,-1,:])
        return y_pred

def preprocess_audio(path):
    _, sr = librosa.load(path)
    raw_audio = AudioSegment.from_file(path)
    
    samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
    trimmed, _ = librosa.effects.trim(samples, top_db=25)
    padded = np.pad(trimmed, (0, 180000-len(trimmed)), 'constant')
    return padded, sr

def get_spec(path):
    FRAME_LENGTH = 2048
    HOP_LENGTH = 512
    y, sr = preprocess_audio(path)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    X = np.concatenate((
        np.swapaxes([zcr], 1, 2), 
        np.swapaxes([rms], 1, 2), 
        np.swapaxes([mfccs], 1, 2)), 
        axis=2
    ).astype('float32')
    return X

def classify(path: str) -> str:
    model = LSTMModel(15, 1024, 1, 6)
    model.load_state_dict(torch.load('LSTM\data\model_best_loss_0.4096_45.pt',map_location=torch.device('cpu')))
    logits = model(torch.tensor(get_spec(path)))
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