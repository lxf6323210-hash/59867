
import torch
import transformers
import sentence_transformers
import accelerate
import PIL
import pandas
import matplotlib

print("All libraries imported successfully!")

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Sentence Transformers:", sentence_transformers.__version__)

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.cuda.is_available())
