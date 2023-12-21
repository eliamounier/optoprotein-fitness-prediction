"""Helper functions for notebooks"""

import pandas as pd
import numpy as np

import re
import csv

from Bio import AlignIO

from Bio.Align import AlignInfo

from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

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
        
def plot_spearman_boxplot(base_path):
    """
    Plot the results as boxplots.

    Parameters:
    - base_path: str, the path to the results you want to plot.

    """
    results = []

    # Iterate over directories
    for folder in os.listdir(base_path):
        if folder.startswith("BL_"):
            path = os.path.join(base_path, folder, "results.csv")
            training_size = int(folder.split('_')[1])  # Extract the training set size

            if os.path.exists(path):
                df = pd.read_csv(path)
                # Check if the expected column exists
                if 'spearman' in df.columns:
                    for index, row in df.iterrows():
                        results.append({'training_size': int(0.2*training_size), 'spearman': row['spearman']})
                else:
                    print(f"'spearman_correlation' column not found in {path}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Plotting
    custom_palette = [((141/255, 211/255, 199/255))]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='training_size', y='spearman', data=results_df, palette=custom_palette)
    plt.xlabel('Test Set Size')
    plt.ylabel('Spearman Correlation')
    plt.savefig('variance_convergence_onehot.pdf', dpi = 300)
    plt.show()

def find_optimal_sequence(light_df, darkness_df, wt_fitness_D=-2.983813):
    """Find the best sequence for double and single mutants"""

    if "Mutated_Position2" in light_df.columns:
        light_df.rename(columns={"pred": "light fitness"}, inplace=True)

        double_mutants = light_df.copy()
        double_mutants["darkness fitness"] = darkness_df["pred"]

        min_all_mut_D = double_mutants["darkness fitness"].min()
        numerator_D = double_mutants["darkness fitness"] - min_all_mut_D + 1
        denominator_D = float(wt_fitness_D - min_all_mut_D + 1)
        threshold = np.log(1000 - numerator_D / denominator_D)

        double_mutants = double_mutants[double_mutants["darkness fitness"] < threshold]
        double_mutants["combined_score"] = (
            double_mutants["light fitness"] - double_mutants["darkness fitness"]
        )
        max_combined_score_index = double_mutants["combined_score"].idxmax()

        optimal_sequence = double_mutants.loc[max_combined_score_index]

        print("Optimal Sequence (Double Mutant):")
        print(
            f"1 Mutated Position: {optimal_sequence['Mutated_Position1']} 2 Mutated Position {optimal_sequence['Mutated_Position2']}"
        )
        print(
            f"Original AA1: {optimal_sequence['Original_AA1']} Original AA2: {optimal_sequence['Original_AA2']}"
        )
        print(
            f"Mutated AA1: {optimal_sequence['Mutated_AA1']} Mutated AA2: {optimal_sequence['Mutated_AA2']}"
        )
        print(f"Sequence: {optimal_sequence['seq']}")
        print(f"Fitness under Light: {optimal_sequence['light fitness']} (Light model)")
        print(
            f"Fitness under Darkness: {optimal_sequence['darkness fitness']} (Darkness model)"
        )
        print(f"Combined Score: {optimal_sequence['combined_score']}")

    else:
        light_df.rename(columns={"pred": "light fitness"}, inplace=True)

        single_mutants = light_df.copy()
        single_mutants["darkness fitness"] = darkness_df["pred"]

        min_all_mut_D = single_mutants["darkness fitness"].min()
        numerator_D = single_mutants["darkness fitness"] - min_all_mut_D + 1
        denominator_D = float(wt_fitness_D - min_all_mut_D + 1)
        threshold = np.log(1000 - numerator_D / denominator_D)

        single_mutants = single_mutants[single_mutants["darkness fitness"] < threshold]
        single_mutants["combined_score"] = (
            single_mutants["light fitness"] - single_mutants["darkness fitness"]
        )

        max_combined_score_index = single_mutants["combined_score"].idxmax()

        optimal_sequence = single_mutants.loc[max_combined_score_index]

        print("Optimal Sequence (Single Mutant):")
        print(f"Mutated Position: {optimal_sequence['Mutated_Position']}")
        print(f"Original AA: {optimal_sequence['Original_AA']}")
        print(f"Mutated AA: {optimal_sequence['Mutated_AA']}")
        print(f"Sequence: {optimal_sequence['seq']}")
        print(f"Fitness under Light: {optimal_sequence['light fitness']} (Light model)")
        print(
            f"Fitness under Darkness: {optimal_sequence['darkness fitness']} (Darkness model)"
        )
        print(f"Combined Score: {optimal_sequence['combined_score']}")

    return optimal_sequence


def calculate_sequence_homology(fasta_file):
    """Finds the sequences homolgy"""
    alignment = AlignIO.read(fasta_file, "fasta")
    num_sequences = len(alignment)
    length_of_alignment = alignment.get_alignment_length()
    homology_count = 0

    for i in range(length_of_alignment):
        column = alignment[:, i]
        if column.count(column[0]) == num_sequences:
            homology_count += 1

    homology_percentage = (homology_count / length_of_alignment) * 100
    return homology_percentage


def convert_to_correct_a2m(a2m_filename, output_filename):
    """
    Processes an A2M file using Biopytho
    """
    alignment = AlignIO.read(a2m_filename, "fasta")

    wild_type_seq = str(alignment[0].seq)
    positions_to_keep = [i for i, char in enumerate(wild_type_seq) if char not in "-."]

    processed_wild_type_seq = "".join(wild_type_seq[i] for i in positions_to_keep)
    processed_wild_type_record = SeqRecord(
        Seq(processed_wild_type_seq), id="Q2NB98", description=""
    )

    new_alignment = MultipleSeqAlignment([processed_wild_type_record])
    for record in alignment[1:]:
        new_seq = "".join(
            str(record.seq)[i] for i in positions_to_keep if i < len(record.seq)
        )
        new_record = SeqRecord(Seq(new_seq), id=record.id, description="")
        new_alignment.append(new_record)

    AlignIO.write(new_alignment, output_filename, "fasta")


def remove_high_gap_sequences(input_file, output_file, gap_threshold=30):
    """
    Remove sequences that have more than 30% gaps.
    """

    alignment = AlignIO.read(input_file, "fasta")
    total_sequences = len(alignment)

    filtered_sequences = [
        record
        for record in alignment
        if ((str(record.seq).count("-") + str(record.seq).count(".")) / len(record.seq))
        * 100
        < gap_threshold
    ]

    filtered_sequences[0].id = "Q2NB98_30perc"
    filtered_sequences[0].name = "Q2NB98_30perc"
    filtered_sequences[0].description = ""

    filtered_alignment = MultipleSeqAlignment(filtered_sequences)

    AlignIO.write(filtered_alignment, output_file, "fasta")

    num_removed = total_sequences - len(filtered_sequences)
    perc_removed = (num_removed / total_sequences) * 100

    return num_removed, perc_removed


def extract_lov_domain_tuples(df):
    """Finds LOV domains in MSA"""
    lov_domain_data = []

    for index, row in df.iterrows():
        if index < 3:
            continue

        protein_id = row.iloc[1]
        domain_info = row.iloc[6]

        print(f"Debug: Protein ID: {protein_id}, Domain Info: {domain_info}")

        match = re.search(r"LOV : \((\d+), (\d+)\)", domain_info)
        if match:
            start, end = match.groups()
            lov_domain_data.append((protein_id, (int(start), int(end))))

    return lov_domain_data
