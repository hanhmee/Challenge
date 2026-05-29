from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

# define function to read in sound file
def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

# load dummy dataset and read soundfiles
ds = map_to_array({
    "file": 'TTD_7_NAM_S0001_78.wav'
})

# tokenize
input_values = processor(ds["speech"], return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
# 1. Đưa dữ liệu qua mô hình (Giả sử bạn đã load mô hình vào biến 'model')
with torch.no_grad():
    logits = model(input_values).logits

# 2. Lấy vị trí phần tử có xác suất cao nhất (Argmax)
predicted_ids = torch.argmax(logits, dim=-1)

# 3. Dịch các ID này thành văn bản (Text)
transcription = processor.batch_decode(predicted_ids)[0]

# 4. In kết quả ra màn hình Terminal
print("\n--- KẾT QUẢ NHẬN DIỆN ---")
print(transcription)
print("-------------------------\n")