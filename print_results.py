import os
import glob
import sys

# Base directory containing all runs
base_dir = "/home/albert/multi-agent-retrieval/all_results_200"

# Metrics to average (base metrics)
base_metrics = ["precision", "recall", "map", "xinfAP", "match_num", "unmatch_num", "unjudge_num"]

# Additional metrics when dataset_name is specified
dataset_metrics = ["vlm_verifications_at_pool_capacity"]

# Parse command line arguments
dataset_name = None
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]

# Determine output filename
if dataset_name:
    output_filename = f'results_output_{dataset_name}.txt'
else:
    output_filename = 'results_output.txt'

# Dual output to console and file
output_file = open(output_filename, 'w')
class DualOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        pass

original_stdout = sys.stdout
sys.stdout = DualOutput(original_stdout, output_file)

# Determine which directories to process
run_dirs = []

# Get all top-level directories in all_results
for top_dir in sorted(os.listdir(base_dir)):
    top_dir_path = os.path.join(base_dir, top_dir)
    if not os.path.isdir(top_dir_path) or top_dir == "_captions":
        continue
    
    if dataset_name:
        # If dataset_name is specified, look for subdirectories with that name
        dataset_subdir_path = os.path.join(top_dir_path, dataset_name)
        if os.path.isdir(dataset_subdir_path):
            # Check if this directory contains eval_results files
            if glob.glob(os.path.join(dataset_subdir_path, "eval_results_*.txt")):
                # Use a combination of top_dir and dataset_name for display
                display_name = f"{top_dir}/{dataset_name}"
                run_dirs.append((display_name, dataset_subdir_path))
    else:
        # Original behavior: process top-level directory
        run_dirs.append((top_dir, top_dir_path))

# Print header information
if dataset_name:
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*80}\n")

# Process each run
for run_name, results_dir in run_dirs:
    # Determine which metrics to use
    if dataset_name:
        metrics = base_metrics + dataset_metrics
    else:
        metrics = base_metrics
    
    # Initialize totals for this run
    totals = {m: 0 for m in metrics}
    count = 0
    
    # Load all eval_results_*.txt files in this run
    for filepath in sorted(glob.glob(os.path.join(results_dir, "eval_results_*.txt"))):
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    if key in metrics:
                        try:
                            totals[key] += float(value.strip())
                        except ValueError:
                            pass
        count += 1
    
    # Calculate and print averages for this run
    if count > 0:
        print(f"\n{'='*80}")
        if dataset_name:
            print(f"Subdataset: {run_name}")
        else:
            print(f"Run: {run_name}")
        print(f"{'='*80}")
        print(f"Average metrics across {count} files:\n")
        for metric in metrics:
            avg = totals[metric] / count
            print(f"{metric}: {avg}")
    else:
        print(f"\nWarning: No eval_results files found in {run_name}")

# Restore stdout and close file
sys.stdout = original_stdout
output_file.close()
print(f"\nResults saved to {output_filename}")
        
