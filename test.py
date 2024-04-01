import transformers
import torch
import accelerate
print(transformers.__version__)
print(torch.__version__)
print(accelerate.__version__)


if torch.cuda.is_available():
    print(f"CUDA is available. GPU Count: {torch.cuda.device_count()}")
    print(f"Current CUDA Device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")



