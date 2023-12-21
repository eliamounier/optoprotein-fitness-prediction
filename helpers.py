"""Helper functions for notebooks"""

import pandas as pd
import numpy as np
import os


def generate_mutations(wt_seq):
    """Generate all possible single mutants given a wild type sequence"""

    mutations = []
    for i in range(len(wt_seq)):
        for amino_acid in "ACDEFGHIKLMNPQRSTVWY":
            if amino_acid != wt_seq[i]:
                mutated_seq = list(wt_seq)
                mutated_seq[i] = amino_acid
                mutated_seq = "".join(mutated_seq)

                mutation = {
                    "Position": i + 1,
                    "Original_AA": wt_seq[i],
                    "Mutated_Position": i + 1,
                    "Mutated_AA": amino_acid,
                    "seq": mutated_seq,
                }
                mutations.append(mutation)
    return mutations


def generate_double_mutants(wt_seq):
    """Generate all possible double mutants given a wild type sequence"""

    double_mutants = []
    seq_length = len(wt_seq)

    for i in range(seq_length):
        for j in range(i + 1, seq_length):
            for amino_acid_i in "ACDEFGHIKLMNPQRSTVWY":
                for amino_acid_j in "ACDEFGHIKLMNPQRSTVWY":
                    if amino_acid_i != wt_seq[i] and amino_acid_j != wt_seq[j]:
                        mutated_seq = list(wt_seq)
                        mutated_seq[i] = amino_acid_i
                        mutated_seq[j] = amino_acid_j
                        mutated_seq = "".join(mutated_seq)

                        double_mutant = {
                            "Position1": i + 1,
                            "Original_AA1": wt_seq[i],
                            "Mutated_Position1": i + 1,
                            "Mutated_AA1": amino_acid_i,
                            "Position2": j + 1,
                            "Original_AA2": wt_seq[j],
                            "Mutated_Position2": j + 1,
                            "Mutated_AA2": amino_acid_j,
                            "seq": mutated_seq,
                        }
                        double_mutants.append(double_mutant)

    return double_mutants


def split_csv(input_csv_path, output_folder, chunk_size):
    """
    Split a CSV file into smaller chunks.

    Parameters:
    - input_csv_path: str, the path to the input CSV file.
    - output_folder: str, the folder where the smaller CSV files will be saved.
    - chunk_size: int, the number of rows in each smaller CSV file.
    """

    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(input_csv_path)

    num_chunks = len(df) // chunk_size

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_df = df.iloc[start_idx:end_idx, :]

        output_csv_path = os.path.join(output_folder, f"chunk_{i + 1}.csv")

        chunk_df.to_csv(output_csv_path, index=False)

    if len(df) % chunk_size != 0:
        last_chunk_df = df.iloc[num_chunks * chunk_size :, :]
        output_csv_path = os.path.join(output_folder, f"chunk_{num_chunks + 1}.csv")
        last_chunk_df.to_csv(output_csv_path, index=False)
