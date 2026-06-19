import kagglehub
import os
import shutil

print("Downloading dataset...")
path = kagglehub.dataset_download("ndayisabae/nf-unsw-nb15-v3")
print("Dataset downloaded to:", path)

# Find the csv file
csv_file = None
for f in os.listdir(path):
    if f.endswith('.csv'):
        csv_file = os.path.join(path, f)
        break

if csv_file:
    # Create target directory
    target_dir = os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, "NF-UNSW-NB15-v3.csv")
    print(f"Moving {csv_file} to {target_path}")
    shutil.copy(csv_file, target_path)
    print("Done!")
else:
    print("Error: Could not find CSV file in downloaded dataset.")
