import qai_hub as hub

# Grab your successfully compiled model using its Job ID
compile_job = hub.get_job("jp0mmoveg") 
target_device = hub.Device("Snapdragon X Elite CRD")

print("Submitting Profile Job to capture NPU hardware benchmarks...")

# Run the compiled model through a performance stress test
profile_job = hub.submit_profile_job(
    model=compile_job.get_target_model(),
    device=target_device,
    name="Faydem CLAP Audio Profiler"
)

print(f"✅ Profiling job submitted! Job ID: {profile_job.job_id}")