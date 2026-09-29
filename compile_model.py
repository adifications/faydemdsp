import qai_hub as hub

print("Connecting to Qualcomm AI Hub...")

# 1. Select the target device (The processor we want to optimize for)
target_device = hub.Device("Snapdragon X Elite CRD")

# 2. Tell the cloud what shape of audio the model expects
# (This model expects: 1 batch, 2 stereo channels, 343,980 samples)
audio_shape = (1, 2, 343980)

print("Uploading model and submitting compile job...")

# 3. Submit the job to the cloud hardware
compile_job = hub.submit_compile_job(
    model="htdemucs_ft_drums.onnx",
    device=target_device,
    name="Demucs NPU Drums Extractor",
    input_specs=dict(mix=audio_shape),
    options="--target_runtime onnx" 
)

print(f"Job submitted successfully! Job ID: {compile_job.job_id}")
print("You can view the progress on the Qualcomm AI Hub website.")

# 4. Download the NPU-optimized file once finished
optimized_model = compile_job.get_target_model()
optimized_model.download("demucs_npu_drums.onnx")
print("Optimized model downloaded!")