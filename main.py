import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

prompt= "नमस्कार! मारुती सुझुकी कस्टमर केअरमध्ये आपले स्वागत आहे! आम्ही सदैव तुमच्या सेवेत आहोत. तुमच्या कारबद्दल कोणतेही प्रश्न किंवा चिंता असल्यास कृपया आमच्याशी संपर्क साधा."
description = "Sanjay speaks in a slighly expressive tone and low pitch, with a slow pace and very clear voice, in a close sounding emvironment."

description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("mr_2.wav", audio_arr, model.config.sampling_rate)