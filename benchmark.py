import torch
import timeit
import torch.nn as nn

from tqdm import tqdm

from vqvae.model import VQVAE

# Define a simple PyTorch model
# Create an instance of the model
model = VQVAE(
    h_dim=128,
    res_h_dim=64,
    n_res_layers=4,
    n_embeddings=1024,
    stacks=3,
    embedding_dim=64,
    beta=0.5,
)

torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create a random input tensor
input_tensor = torch.randn((32, 3, 256, 256), device=device)
input_tensor = input_tensor.to(memory_format=torch.channels_last)

model = model.to(device)
model = model.to(memory_format=torch.channels_last)

model = torch.compile(model, mode='max-autotune', fullgraph=True)


# Define the function to measure
def forward_pass():
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        model(input_tensor)


# Set up the timeit parameters
number = 200  # Number of executions per measurement
repeat = 5  # Number of measurements

# warmup pass
with torch.no_grad():
    model(input_tensor)

# Run the benchmark
results = timeit.repeat(forward_pass, number=number, repeat=repeat)

# Print the results
print(f'Forward pass time (average of {repeat} runs):')
print(f'Min: {min(results) / number:.6f} seconds')
print(f'Max: {max(results) / number:.6f} seconds')
print(f'Avg: {sum(results) / (number * repeat):.6f} seconds')
