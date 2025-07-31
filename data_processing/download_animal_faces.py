import kagglehub

# Download latest version
path = kagglehub.dataset_download("hjjiao/animals-face-256")

print("Path to dataset files:", path)