import torch
import qai_hub as hub
from transformers import ClapModel, ClapProcessor
import numpy as np
import warnings

warnings.filterwarnings("ignore")

print("1. Loading CLAP architecture from Hugging Face...")
MODEL_ID = "laion/clap-htsat-unfused"
processor = ClapProcessor.from_pretrained(MODEL_ID)
base_model = ClapModel.from_pretrained(MODEL_ID)

# Strip out everything except the Audio Encoder
class CLAPAudioEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.audio_model = model.audio_model
        
    def forward(self, input_features):
        return self.audio_model(input_features).pooler_output

encoder_wrapper = CLAPAudioEncoder(base_model)
encoder_wrapper.eval()

print("2. Generating dummy spectrogram for PyTorch tracing...")
dummy_audio = np.zeros(48000 * 5) 
inputs = processor(audio=dummy_audio, sampling_rate=48000, return_tensors="pt")
dummy_input = inputs["input_features"]
print(f"   Expected NPU Spectrogram Shape: {tuple(dummy_input.shape)}")

print("3. Tracing model directly to Qualcomm API...")
# Tracing the model ensures all external weights are packaged automatically by qai_hub
traced_model = torch.jit.trace(encoder_wrapper, dummy_input)

print("4. Connecting to Qualcomm Cloud Labs (Snapdragon X Elite)...")
target_device = hub.Device("Snapdragon X Elite CRD")

print("   Submitting NPU compilation job...")
compile_job = hub.submit_compile_job(
    model=traced_model,
    device=target_device,
    input_specs=dict(input_features=tuple(dummy_input.shape)),
    options="--target_runtime onnx",
    name="Faydem CLAP Audio Encoder"
)

print(f"✅ Job submitted successfully! Job ID: {compile_job.job_id}")
print("   Waiting for physical device compilation (this may take 5-10 minutes)...")

optimized_model = compile_job.get_target_model()
optimized_model.download("clap_audio_npu.onnx")
print("✅ Snapdragon NPU Optimized Model Saved: clap_audio_npu.onnx")