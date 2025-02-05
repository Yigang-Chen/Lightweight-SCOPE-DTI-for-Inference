"""
This script performs batch inference using the light weight version SCOPE model for Drug-Target Interaction (DTI) prediction.
The main steps involved are:
1. Input Data Preprocessing: Generating SDF files from given compound identifiers (CID, SMILES, or names).
2. Prediction File Generation: Creating prediction files for each compound using a pre-defined protein structure template.
3. Model Loading: Loading pre-trained SCOPE models for different target types (e.g., GPCR, IC).
4. Inference: Performing predictions using the loaded models and saving the results.
5. Cleanup: Removing intermediate files and folders after inference is complete.

Example command to run the script:
1.python inference_new.py --model_dir models_path --output_dir output_predictions_1021 --input_type name --input_values canagliflozin
2.python inference_new.py --model_dir models_path --output_dir output_predictions_1022 --input_type cid --input_values Nodes.xlsx
"""

import torch
import os
import pandas as pd
from models import SCOPE
from utils import custom_collate_fn, mkdir, CompoundSDFGenerator, PredictFileGenerator
from dataloader_new import DTIDataset
from torch_geometric.loader import DataLoader  # Using PyTorch Geometric DataLoader
# from compound_sdf_generator import CompoundSDFGenerator
# from predict_file_generator import PredictFileGenerator
from predictor import InferenceHandler
from configs import get_cfg_defaults
import argparse
import shutil

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse command line arguments
parser = argparse.ArgumentParser(description="Batch Inference using SCOPE model for DTI prediction")

parser.add_argument('--model_dir', required=True, help="Path to the model directory", type=str)
parser.add_argument('--output_dir', required=True, help="Output directory for predictions", type=str)
parser.add_argument('--input_type', required=True, choices=["cid", "smiles", "name"],
                    help="Input type: cid, smiles or name")
parser.add_argument('--input_values', nargs='+',
                    help="List of input values (e.g., CID numbers, SMILES strings, or compound names) or a file containing them",
                    required=True)
args = parser.parse_args()


def load_models(model_dir):
    models = {}
    # Iterate through each model type (GPCR, IC, etc.)
    for model_type in ['GPCR', 'IC', 'Kinase', 'NHR', 'Total']:
        model_path = os.path.join(model_dir, f'Filtered_{model_type}_DrugBAN_3D')
        models[model_type] = []
        for model_file in sorted(os.listdir(model_path)):
            if model_file.endswith('.pth'):
                models[model_type].append(os.path.join(model_path, model_file))
    return models


def inference_main():
    print("Starting input data preprocessing...")
    torch.cuda.empty_cache()  # Clear GPU cache
    cfg = get_cfg_defaults()  # Get default config

    # Set prediction output directory
    Output_dir = args.output_dir
    Intermediate_dir = 'intermediate'  # Set intermediate directory
    mkdir(Output_dir)  # Create output directory
    mkdir(Intermediate_dir)  # Create intermediate directory

    # Step 1: Generate SDF files using CompoundSDFGenerator
    sdf_generator = CompoundSDFGenerator()
    df = None
    if len(args.input_values) == 1 and os.path.isfile(args.input_values[0]):
        # If input is a file, read file content
        input_file = args.input_values[0]
        df = pd.read_excel(input_file)

        # Convert all column names to lowercase
        df.columns = df.columns.str.lower()
        if 'sdf' in df.columns and 'compound' in df.columns:
            # If 'sdf' column is present, use it directly
            sdf_list = df[['sdf', 'compound']].dropna().values.tolist()
            sdf_output_file = os.path.join(Intermediate_dir, "compounds_sdf_from_file.xlsx")
            sdf_df = pd.DataFrame(sdf_list, columns=['sdf', 'compound'])
            sdf_df.to_excel(sdf_output_file, index=False)
            print("Using provided SDF data from input file.")
        elif 'smiles' in df.columns and 'compound' in df.columns:
            input_values = df[['smiles', 'compound']].dropna().values.tolist()
            smiles_list = [item[0] for item in input_values]
            compound_names = [item[1] for item in input_values]
        elif 'compound' in df.columns:
            # If only compound column is available, use compound as both name and input value
            compound_names = df['compound'].dropna().tolist()
            smiles_list = compound_names
        else:
            print("No valid column (e.g., 'sdf', 'smiles', 'compound') found in the input file.")
            exit(1)
    else:
        # If input is direct values
        input_values = args.input_values
        compound_names = input_values
        smiles_list = input_values

    # If 'sdf' was not directly provided, generate SDF using SMILES list
    if df is None or 'sdf' not in df.columns:
    # Generate SDF using SMILES list, and keep the compound names for file naming
        sdf_output_file = os.path.join(Intermediate_dir, "compounds_sdf.xlsx")
        sdf_generator.generate_sdf_from_input(args.input_type, smiles_list, compound_names, sdf_output_file)

    # Step 1.1: Verify the generated SDF files
    sdf_df = pd.read_excel(sdf_output_file)
    sdf_df.columns = sdf_df.columns.str.lower()
    invalid_compounds = []

    # Assuming that an empty SDF will have a NaN or empty value in the 'sdf' column
    if 'sdf' in sdf_df.columns:
        invalid_sdf_entries = sdf_df[sdf_df['sdf'].isna() | (sdf_df['sdf'] == '')]
        invalid_compounds = invalid_sdf_entries['compound'].tolist()

        # Filter out invalid compounds from the SDF dataframe
        sdf_df = sdf_df[~sdf_df['compound'].isin(invalid_compounds)]
        sdf_df.to_excel(sdf_output_file, index=False)  # Save the filtered SDF data
    print(f"Invalid compounds with empty SDFs: {invalid_compounds}")
    # Step 2: Generate prediction files using PredictFileGenerator
    predict_generator = PredictFileGenerator(sdf_output_file, 'protein_targets', Intermediate_dir)
    predict_generator.load_input_data()
    predict_generator.setup_output_directory()
    predict_generator.load_protein_data()
    predict_generator.generate_predict_files()

    print("Data preprocessing completed. Starting inference...")
    # Step 3: Load models for inference
    models = load_models(args.model_dir)

    # Step 4: Perform inference
    for compound in os.listdir(Intermediate_dir):
        compound_dir = os.path.join(Intermediate_dir, compound)
        if not os.path.isdir(compound_dir):
            continue

        # Iterate through each prediction file in the compound folder
        for predict_file in os.listdir(compound_dir):
            if predict_file.endswith('_predict.parquet'):
                model_type = predict_file.split('_')[0]  # Extract model type (e.g., GPCR, IC)
                model_paths = models.get(model_type)

                if not model_paths:
                    print(f"No models found for {model_type}")
                    continue

                # Load data
                test_path = os.path.join(compound_dir, predict_file)
                df_test = pd.read_parquet(test_path)
                df_test['label'] = 0  # Add temporary label column

                # Create dataset and DataLoader
                test_dataset = DTIDataset(df_test.index.values, df_test)
                params = {
                    'batch_size': cfg.SOLVER.BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': cfg.SOLVER.NUM_WORKERS,
                    'drop_last': False,
                    'pin_memory': True,
                    'collate_fn': custom_collate_fn
                }
                test_generator = DataLoader(test_dataset, **params)

                # Perform inference for each model path
                for i, model_path in enumerate(model_paths, start=1):
                    # Load model
                    model = SCOPE(**cfg).to(device)
                    trainer = InferenceHandler(model, device, **cfg)
                    trainer.load_model(model_path)

                    # Run inference
                    #predictions, _ = trainer.inference(test_generator)
                    predictions, att_array = trainer.inference(test_generator)  # 现在返回的是predictions和att_array
                    # Add predictions to dataframe
                    df_test[f'predictions_{i}'] = predictions
                    #df_test[f'att_{i}'] = att_array  # 添加att_array列到df_test中

                # Remove temporary label column
                df_test = df_test.drop(columns=['label', 'sequence', 'sdf'])
                df_test['predict_average'] = df_test[['predictions_1','predictions_2','predictions_3','predictions_4','predictions_5']].mean(axis=1)
                df_test = df_test.sort_values(by='predict_average', ascending=False)

                # Save results to output directory
                compound_folder = os.path.join(Output_dir, compound)
                if not os.path.exists(compound_folder):
                    os.makedirs(compound_folder)

                output_file = os.path.join(compound_folder, f"{compound}_{model_type}_predictions.csv")
                df_test.to_csv(output_file, index=False)

                print(f"Inference completed for {compound} - {model_type}, results saved to {output_file}")
    print("Inference completed for all compounds. Cleaning up intermediate files...")

    # Step 5: Cleanup intermediate files and folders
    if os.path.exists(Intermediate_dir):
        shutil.rmtree(Intermediate_dir)
        print(f"Intermediate directory {Intermediate_dir} has been deleted.")

    # Print the list of compounds that failed to generate valid SDF files
    if invalid_compounds:
        print("The following compounds failed to generate valid SDF files:")
        for compound in invalid_compounds:
            print(compound)


if __name__ == '__main__':
    inference_main()
