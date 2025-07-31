from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1='path/to/your/real_images/',
    input2='../data_processing/output',
    cuda=True,
    fid=True,
    verbose=True,
)

print("FID:", metrics['frechet_inception_distance'])
