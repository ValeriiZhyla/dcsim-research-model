import torch

cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
print("PyTorch version: ", torch.__version__)
if cuda_available:
    print("Number of GPUs available:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("GPU", i, ":", torch.cuda.get_device_name(i))