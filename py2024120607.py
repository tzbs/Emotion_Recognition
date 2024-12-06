import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# 音频文件路径
audio_file = "This_is_wonderful.wav"

# 加载音频文件并保持原采样率
audio, sr = librosa.load(audio_file, sr=None)

# 如果原采样率不是16kHz，重采样为16kHz
target_sr = 16000
if sr != target_sr:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

# 使用 Hugging Face 加载 wav2vec2 处理器和模型
model_path = 'E:\\Emotion_Recognition\\the_Wav2Vec2-Large-Robust_model\\wav2vec2-base'
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2Model.from_pretrained(model_path)

# 将音频转换为模型输入所需的格式
input_values = processor(audio, return_tensors="pt", sampling_rate=target_sr).input_values

# 提取音频特征
with torch.no_grad():
    outputs = model(input_values)
    last_hidden_state = outputs.last_hidden_state

# last_hidden_state 是特征矩阵，维度为 [batch_size, sequence_length, feature_dim]
print("Extracted features shape:", last_hidden_state.shape)

# 如果你需要将这些特征保存到文件或者进一步处理，可以根据需求进行操作
