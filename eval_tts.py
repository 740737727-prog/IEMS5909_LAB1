import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

# model = Qwen3TTSModel.from_pretrained(
#     "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
#     device_map="cuda:0",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
# )

model = Qwen3TTSModel.from_pretrained(
    "opt/models/Qwen3-TTS-12Hz-0.6B-Base",
    dtype=dtype,
    device_map=device
)

ref_audio = "my_video.wav"
ref_text  = "Wealth, fame, power, the man who had acquired everything in the world, the parrot king gold Roger, the final words that were said at his execution, sent the people to the seas"

wavs, sr = model.generate_voice_clone(
    text="""To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take Arms against a Sea of troubles,
And by opposing end them: to die, to sleep"""
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("output_voice_clone.wav", wavs[0], sr)
