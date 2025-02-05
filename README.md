# Lightweight SCOPE-DTI for Inference

This repository provides a **lightweight version** of the SCOPE-DTI model, focused primarily on **inference** tasks. It aims to offer performance approximating the **full SCOPE-DTI model** while requiring **significantly less GPU memory** and providing **faster inference speed**.

## Environment Setup

To set up the environment, please follow the steps below using **conda** or **micromamba**:

micromamba create --name scope_inference python=3.9
micromamba activate scope_inference

micromamba install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
micromamba install pyg=2.5.2 -c pyg
micromamba install -c dglteam/label/th22_cu118 dgl
micromamba install -c conda-forge rdkit==2024.03.5
micromamba install pyarrow
micromamba install openpyxl
micromamba install numpy=1

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install dgllife==0.3.2
pip install yacs
pip install pubchempy


## Usage Instructions

1. **Clone or download** this repository locally.
2. **Install** the dependencies in your conda/micromamba environment according to the instructions above.
3. **Run** inference using the provided `inference_new.py` script.

### Example Commands

You can perform inference on:
- A **single compound** or list of **compound1+compound2+compoundx** by passing a single value or a list of values.
- A **batch of compounds** by passing an Excel file (`.xlsx`) containing multiple entries.

Examples:

# Example 1: Inference on a single compound (by name)
python inference_new.py --model_dir models_path \
                        --output_dir output_predictions \
                        --input_type name \
                        --input_values canagliflozin

# Example 2: Inference on a list of compounds (Excel file)
python inference_new.py --model_dir models_path \
                        --output_dir output_predictions \
                        --input_type cid \
                        --input_values Nodes.xlsx

### Input Arguments

| Argument         | Required | Type   | Choices           | Description                                                                  |
|------------------|----------|--------|-------------------|------------------------------------------------------------------------------|
| `--model_dir`    | Yes      | str    | models_path       | Path to the directory containing trained models.                           |
| `--output_dir`   | Yes      | str    | -                 | Directory to save prediction results.                                        |
| `--input_type`   | Yes      | str    | cid, smiles, name | Type of input identifiers.                                                   |
| `--input_values` | Yes      | str(s) | -                 | Input values (e.g., canagliflozin) or an Excel file (e.g., Nodes.xlsx).       |


## Input File Requirements

- **File Format:** Must be an Excel file (`.xlsx`).

- **Required Columns:**
  - A column named `compound` containing either CIDs or compound names.
  - *(Optional)* A `smiles` column if using `--input_type smiles` (recommended for direct SMILES-based inference).

- **Example Excel Structure:**

  | compound      | smiles         |
  |---------------|----------------|
  | canagliflozin | CC1=C(C=C(C=C1)...  |
  | ...           | ...            |

- **Recommendations:**
  - Use the `smiles` column for inference when possible (specify `--input_type smiles`).
  - The `compound` column can store human-readable names for reference.

- **Notes:**
  - For batch inference, ensure input Excel files follow the column specifications above.
  - GPU memory usage and inference speed improvements are benchmarked against the full SCOPE-DTI model.
