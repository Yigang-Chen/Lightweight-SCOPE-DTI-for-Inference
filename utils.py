import os
import numpy as np
import torch
import logging
import torch_geometric
import pandas as pd
import pubchempy as pcp
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from concurrent.futures import ThreadPoolExecutor
import time
import pyarrow.parquet as pq
import pyarrow as pa

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25

def custom_collate_fn(batch):
    # Unpack the batch
    drug_graphs, proteins, labels = zip(*batch)

    # Batch the drug graphs
    drug_batch = torch_geometric.data.Batch.from_data_list(drug_graphs)

    # Stack proteins and labels into tensors
    protein_batch = torch.stack(proteins)
    label_batch = torch.stack(labels)

    return drug_batch, protein_batch, label_batch

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)

def integer_label_protein(sequence, max_length=2000):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding

class CompoundSDFGenerator:
    def __init__(self):
        pass

    # Retry to get the compound's SMILES
    def get_smiles_with_retry(self, id, retries=3, delay=5):
        for attempt in range(retries):
            try:
                compound = pcp.Compound.from_cid(id)
                return compound.canonical_smiles
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for Node {id}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)  # Wait and retry
        return None

    # Generate SDF content from SMILES
    def generate_sdf(self, smiles):
        try:
            # Use RDKit to read SMILES and generate molecule object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Add hydrogen atoms and perform MMFF force field optimization
            mol = Chem.AddHs(mol)

            # AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            # AllChem.MMFFOptimizeMolecule(mol)

            # Try to embed molecule with better parameters
            params = AllChem.ETKDGv3()
            params.maxAttempts = 2000
            params.pruneRmsThresh = 0.1
            if AllChem.EmbedMolecule(mol, params) != 0:
                raise ValueError("Embedding failed, unable to generate conformer")

            # Optimize the molecule using MMFF or UFF
            if not AllChem.MMFFOptimizeMolecule(mol, maxIters=500):
                AllChem.UFFOptimizeMolecule(mol)

            # Generate SDF content
            sdf_content = Chem.MolToMolBlock(mol)
            return sdf_content
        except Exception as e:
            print(f"Error generating SDF for SMILES {smiles}: {e}")
            return None

    # Get SDF content by compound name
    def get_sdf_by_name(self, compound_name):
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/cids/TXT"
        response = requests.get(search_url)

        if response.status_code == 200:
            cid = response.text.strip()
        else:
            print(f"Unable to find CID for {compound_name}.")
            return None

        # Use CID to get SDF content(delete)
        # Use CID to get SMILES and then generate SDF
        return self.process_input("cid", cid)

    # Generate SDF content based on input type
    def process_input(self, input_type, input_value):
        if input_type == "cid":
            # # Use CID to get SDF data
            # sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{input_value}/SDF"
            # sdf_response = requests.get(sdf_url)
            #
            # if sdf_response.status_code == 200:
            #     return sdf_response.text
            # else:
            #     print(f"Unable to retrieve SDF data, attempting to get SMILES for CID: {input_value}")
            #     smiles = self.get_smiles_with_retry(input_value)
            #     if smiles:
            #         return self.generate_sdf(smiles)
            #     else:
            #         print(f"Error generating SDF for SMILES from CID {input_value}")
            #         return None
            # Use CID to get SMILES and then generate SDF
            smiles = self.get_smiles_with_retry(input_value)
            if smiles:
                return self.generate_sdf(smiles)
            else:
                print(f"Error generating SDF for SMILES from CID {input_value}")
                return None
        elif input_type == "smiles":
            return self.generate_sdf(input_value)
        elif input_type == "name":
            return self.get_sdf_by_name(input_value)
        else:
            print("Unsupported input type. Please enter 'cid', 'smiles', or 'name'")
            return None

    # Save SDF content to Excel
    def save_sdf_to_excel(self, compounds, sdfs, filename):
        df = pd.DataFrame({'compound': compounds, 'sdf': sdfs})
        df.to_excel(filename, index=False)
        #print(f"SDF data has been saved to Excel file {filename}.")

    # Main function to process input and save to Excel
    # def generate_sdf_from_input(self, input_type, input_values, output_filename):
    #     compounds = []
    #     sdfs = []
    #     with ThreadPoolExecutor(max_workers=5) as executor:
    #         futures = {executor.submit(self.process_input, input_type, value): value for value in input_values}
    #         for future in futures:
    #             input_value = futures[future]
    #             sdf_content = future.result()
    #             compounds.append(input_value)
    #             sdfs.append(sdf_content if sdf_content else "")
    #
    #     # Save results to Excel
    #     self.save_sdf_to_excel(compounds, sdfs, output_filename)

    # Main function to process input and save to Excel
    def generate_sdf_from_input(self, input_type, smiles_list, compound_names, output_filename):
        compounds = []
        sdfs = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.process_input, input_type, value): value for value in smiles_list}
            for future in futures:
                input_value = futures[future]
                sdf_content = future.result()
                compounds.append(compound_names[smiles_list.index(input_value)])  # Use compound name
                sdfs.append(sdf_content if sdf_content else "")

        # Save results to Excel
        self.save_sdf_to_excel(compounds, sdfs, output_filename)

class PredictFileGenerator:
    def __init__(self, input_excel, input_dir, output_dir):
        self.input_excel = input_excel
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.protein_data = {}
        self.df = None

    def load_input_data(self):
        # Read Excel file
        self.df = pd.read_excel(self.input_excel)

    def setup_output_directory(self):
        # Setup output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_protein_data(self):
        # Select a sample drug folder to get protein structure and information
        sample_drug_folder = os.path.join(self.input_dir)
        if not os.path.exists(sample_drug_folder):
            raise Exception(f"Sample drug folder {sample_drug_folder} does not exist.")

        # Read parquet files in the sample drug folder to get protein information
        for file in os.listdir(sample_drug_folder):
            if file.endswith('_predict.parquet'):
                original_file_path = os.path.join(sample_drug_folder, file)
                df_parquet = pd.read_parquet(original_file_path)
                # Save protein information (excluding sdf column)
                self.protein_data[file] = df_parquet

    def generate_predict_files(self):
        # Iterate through each node and generate corresponding folders and parquet files
        for index, row in self.df.iterrows():
            drug_name = str(row['compound'])
            sdf_content = row['sdf']

            # Create drug folder
            drug_folder_path = os.path.join(self.output_dir, drug_name)
            if not os.path.exists(drug_folder_path):
                os.makedirs(drug_folder_path)

            # Use the protein information from the sample drug, update the sdf column with the current drug's sdf content
            for file_name, protein_df in self.protein_data.items():
                # Create a copy and add sdf column
                updated_df = protein_df.copy()
                updated_df['sdf'] = sdf_content

                # Construct output parquet file path
                target_file_path = os.path.join(drug_folder_path, file_name)

                # Save the updated DataFrame as a Parquet file
                table = pa.Table.from_pandas(updated_df)
                pq.write_table(table, target_file_path)

        #print("All Parquet files have been generated and updated.")
