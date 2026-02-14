#!/usr/bin/env python3
"""
Script to generate a LaTeX table from result files.
Extracts map and vlm_verifications_at_pool_capacity metrics from:
- Probabilistic models with varying confidence levels
- Multiagent baseline model

Output: result_summary.tex
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_result_file(filepath: str) -> List[Dict]:
    """
    Parse a result file and extract runs with their metrics.
    
    Returns a list of dicts with keys:
    - run_name: Full run name
    - model_type: 'probabilistic' or 'multiagent_baseline'
    - confidence: Confidence value (for probabilistic models)
    - dataset: Dataset name (from file or subdataset header)
    - map: Mean Average Precision
    - vlm_calls: VLM verifications at pool capacity
    """
    runs = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    dataset_name = extract_dataset_name(filepath)
    
    # Split by run headers
    run_pattern = r'=+\s*\nRun:\s*(.+?)\n=+\s*\n(.+?)(?==+\s*\nRun:|$)'
    run_matches = re.finditer(run_pattern, content, re.DOTALL)
    
    # Process run headers
    for match in run_matches:
        run_name = match.group(1).strip()
        run_content = match.group(2)
        
        run_data = parse_run_content(run_name, run_content, dataset_name)
        if run_data:
            runs.append(run_data)
    
    # Also try subdataset pattern for v3c2 and iacc.3 files
    subdataset_pattern = r'=+\s*\nSubdataset:\s*(.+?)\n=+\s*\n(.+?)(?==+\s*\nSubdataset:|$)'
    subdataset_matches = re.finditer(subdataset_pattern, content, re.DOTALL)
    
    # Process subdataset headers
    for match in subdataset_matches:
        run_name = match.group(1).strip()
        run_content = match.group(2)
        
        # Extract dataset from subdataset path
        if '/' in run_name:
            subdataset_parts = run_name.split('/')
            dataset_from_subdataset = subdataset_parts[-1]
        else:
            dataset_from_subdataset = dataset_name
        
        run_data = parse_run_content(run_name, run_content, dataset_from_subdataset)
        if run_data:
            runs.append(run_data)
    
    return runs


def extract_dataset_name(filepath: str) -> str:
    """Extract dataset name from filepath."""
    if 'v3c1' in filepath:
        return 'v3c1'
    elif 'v3c2' in filepath:
        return 'v3c2'
    elif 'iacc' in filepath:
        return 'iacc.3'
    return 'unknown'


def parse_run_content(run_name: str, content: str, dataset: str) -> Dict:
    """Parse metrics from run content."""
    
    # Extract map value
    map_match = re.search(r'map:\s*([\d.]+)', content)
    if not map_match:
        return None
    
    map_value = float(map_match.group(1))
    
    # Extract vlm_verifications_at_pool_capacity
    vlm_match = re.search(r'vlm_verifications_at_pool_capacity:\s*([\d.]+)', content)
    if not vlm_match:
        return None
    
    vlm_value = float(vlm_match.group(1))
    
    # Determine model type and extract confidence if applicable
    if 'ViSA_probabilistic_policy__MLLM_Qwen2.5-VL-7B-Instruct_search_model_viclip_eval_k_100_examine_number_20_MAX_ITER_200_confidence_' in run_name:
        model_type = 'probabilistic'
        # Extract confidence value
        confidence_match = re.search(r'confidence_([\d.]+)', run_name)
        if confidence_match:
            confidence = float(confidence_match.group(1))
        else:
            return None
    elif 'ViSA_zero_shot__MLLM_Qwen2.5-VL-7B-Instruct_search_model_viclip_eval_k_100_examine_number_20_MAX_ITER_200_action_type_reasoning_reformulation_type_with_action_reasoning' in run_name:
        model_type = 'multiagent_baseline'
        confidence = None
    elif 'Naive_Retrieval__MLLM_Qwen2.5-VL-7B-Instruct_search_model_viclip_eval_k_100' in run_name:
        model_type = 'naive'
        confidence = None
    else:
        return None
    
    return {
        'run_name': run_name,
        'model_type': model_type,
        'confidence': confidence,
        'dataset': dataset,
        'map': map_value,
        'vlm_calls': vlm_value
    }


def organize_data(runs: List[Dict]) -> Dict:
    """
    Organize runs by model type and dataset.
    
    Returns:
    {
        'v3c1': {
            'probabilistic': {conf_val: {'map': ..., 'vlm_calls': ...}},
            'multiagent_baseline': {'map': ..., 'vlm_calls': ...},
            'naive': {'map': ..., 'vlm_calls': ...}
        },
        ...
    }
    """
    organized = {}
    
    for run in runs:
        dataset = run['dataset']
        model_type = run['model_type']
        
        if dataset not in organized:
            organized[dataset] = {
                'probabilistic': {},
                'multiagent_baseline': {},
                'naive': {}
            }
        
        if model_type == 'probabilistic':
            conf = run['confidence']
            organized[dataset]['probabilistic'][conf] = {
                'map': run['map'],
                'vlm_calls': run['vlm_calls']
            }
        elif model_type == 'multiagent_baseline':
            organized[dataset]['multiagent_baseline'] = {
                'map': run['map'],
                'vlm_calls': run['vlm_calls']
            }
        elif model_type == 'naive':
            organized[dataset]['naive'] = {
                'map': run['map'],
                'vlm_calls': run['vlm_calls']
            }
    
    return organized


def generate_latex_table(organized: Dict) -> str:
    """Generate LaTeX table from organized data."""
    
    # Determine datasets in order
    datasets = sorted(organized.keys())
    # Reorder to match the typical order
    dataset_order = ['v3c1', 'v3c2', 'iacc.3']
    datasets = [d for d in dataset_order if d in datasets]
    
    # Collect all confidence values from all datasets
    all_confidences = set()
    for dataset in datasets:
        all_confidences.update(organized[dataset]['probabilistic'].keys())
    
    confidences = sorted(all_confidences)
    
    # Find best values (highest MAP, lowest VLM calls) for each dataset
    best_map_per_dataset = {}
    best_vlm_per_dataset = {}
    
    for dataset in datasets:
        map_values = []
        vlm_values = []
        
        # Collect from probabilistic models
        for conf in confidences:
            if conf in organized[dataset]['probabilistic']:
                data = organized[dataset]['probabilistic'][conf]
                map_values.append(data['map'])
                vlm_values.append(data['vlm_calls'])
        
        # Collect from multiagent baseline
        if 'multiagent_baseline' in organized[dataset] and organized[dataset]['multiagent_baseline']:
            data = organized[dataset]['multiagent_baseline']
            map_values.append(data['map'])
            vlm_values.append(data['vlm_calls'])
        
        # Collect from naive baseline
        if 'naive' in organized[dataset] and organized[dataset]['naive']:
            data = organized[dataset]['naive']
            map_values.append(data['map'])
            vlm_values.append(data['vlm_calls'])
        
        if map_values:
            best_map_per_dataset[dataset] = max(map_values)
        if vlm_values:
            best_vlm_per_dataset[dataset] = min(vlm_values)
    
    # Build LaTeX table
    latex = []
    latex.append(r'\begin{table}[htbp]')
    latex.append(r'\centering')
    latex.append(r'\caption{Comparison of ViSA Models on Different Datasets}')
    latex.append(r'\label{tab:results}')
    latex.append('')
    
    # Column definition: left column for row names + 6 columns for 3 datasets × 2 metrics
    latex.append(r'\begin{tabular}{|l|cc|cc|cc|}')
    latex.append(r'\hline')
    
    # Header row 1: Dataset names
    dataset_headers = ' & '.join([r'\multicolumn{2}{c|}{\textbf{' + d + '}}' for d in datasets])
    header1 = r'\textbf{Model} & ' + dataset_headers + r' \\'
    latex.append(header1)
    
    # Header row 2: Metric names
    metric_header = ' & '.join([r'\textbf{MAP} & \textbf{VLM Calls}'] * len(datasets))
    header2 = r' & ' + metric_header + r' \\'
    latex.append(header2)
    latex.append(r'\hline')
    
    # Data rows for probabilistic models
    for conf in confidences:
        row_name = f'Probabilistic {conf}'
        row = [row_name]
        
        for dataset in datasets:
            if conf in organized[dataset]['probabilistic']:
                data = organized[dataset]['probabilistic'][conf]
                
                # Format MAP with bold if it's the best
                map_val = data['map']
                map_str = f"{map_val:.4f}"
                if dataset in best_map_per_dataset and abs(map_val - best_map_per_dataset[dataset]) < 1e-6:
                    map_str = r'\textbf{' + map_str + '}'
                
                # Format VLM calls with bold if it's the best
                vlm_val = data['vlm_calls']
                vlm_str = f"{vlm_val:.2f}"
                if dataset in best_vlm_per_dataset and abs(vlm_val - best_vlm_per_dataset[dataset]) < 1e-6:
                    vlm_str = r'\textbf{' + vlm_str + '}'
                
                row.append(map_str)
                row.append(vlm_str)
            else:
                row.append('---')
                row.append('---')
        
        latex.append(' & '.join(row) + r' \\')
    
    # Data row for multiagent baseline
    latex.append(r'\hline')
    row_name = r'\textit{Multiagent Baseline}'
    row = [row_name]
    
    for dataset in datasets:
        if 'multiagent_baseline' in organized[dataset] and organized[dataset]['multiagent_baseline']:
            data = organized[dataset]['multiagent_baseline']
            
            # Format MAP with bold if it's the best
            map_val = data['map']
            map_str = f"{map_val:.4f}"
            if dataset in best_map_per_dataset and abs(map_val - best_map_per_dataset[dataset]) < 1e-6:
                map_str = r'\textbf{' + map_str + '}'
            
            # Format VLM calls with bold if it's the best
            vlm_val = data['vlm_calls']
            vlm_str = f"{vlm_val:.2f}"
            if dataset in best_vlm_per_dataset and abs(vlm_val - best_vlm_per_dataset[dataset]) < 1e-6:
                vlm_str = r'\textbf{' + vlm_str + '}'
            
            row.append(map_str)
            row.append(vlm_str)
        else:
            row.append('---')
            row.append('---')
    
    latex.append(' & '.join(row) + r' \\')
    
    # Data row for naive baseline
    row_name = r'\textit{Naive}'
    row = [row_name]
    
    for dataset in datasets:
        if 'naive' in organized[dataset] and organized[dataset]['naive']:
            data = organized[dataset]['naive']
            
            # Format MAP with bold if it's the best
            map_val = data['map']
            map_str = f"{map_val:.4f}"
            if dataset in best_map_per_dataset and abs(map_val - best_map_per_dataset[dataset]) < 1e-6:
                map_str = r'\textbf{' + map_str + '}'
            
            # Format VLM calls with bold if it's the best
            vlm_val = data['vlm_calls']
            vlm_str = f"{vlm_val:.2f}"
            if dataset in best_vlm_per_dataset and abs(vlm_val - best_vlm_per_dataset[dataset]) < 1e-6:
                vlm_str = r'\textbf{' + vlm_str + '}'
            
            row.append(map_str)
            row.append(vlm_str)
        else:
            row.append('---')
            row.append('---')
    
    latex.append(' & '.join(row) + r' \\')
    
    latex.append(r'\hline')
    latex.append(r'\end{tabular}')
    latex.append('')
    latex.append(r'\end{table}')
    
    return '\n'.join(latex)


def main():
    """Main execution."""
    
    # File paths
    files = [
        'results_output_v3c1.txt',
        'results_output_v3c2.txt',
        'results_output_iacc.3.txt'
    ]
    
    # Parse all files
    all_runs = []
    for filepath in files:
        if Path(filepath).exists():
            print(f"Parsing {filepath}...")
            runs = parse_result_file(filepath)
            all_runs.extend(runs)
            print(f"  Found {len(runs)} runs")
        else:
            print(f"Warning: {filepath} not found")
    
    # Organize data
    organized = organize_data(all_runs)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(organized)
    
    # Write to file
    output_file = 'result_summary.tex'
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table written to {output_file}")
    
    # Print summary
    print("\nData summary:")
    for dataset in sorted(organized.keys()):
        print(f"\n{dataset}:")
        print(f"  Probabilistic models: {len(organized[dataset]['probabilistic'])} confidence levels")
        if organized[dataset]['multiagent_baseline']:
            print(f"  Multiagent baseline: ✓")
        else:
            print(f"  Multiagent baseline: ✗ (not found)")
        if organized[dataset]['naive']:
            print(f"  Naive baseline: ✓")
        else:
            print(f"  Naive baseline: ✗ (not found)")


if __name__ == '__main__':
    main()
