import argparse
import numpy as np
import os
import pandas as pd
import re


def main():
    parser = argparse.ArgumentParser(description="Creates csv files containing RNAseq data")
    parser.add_argument("expression_file",
                        help="The path to the tsv file containing the RNA expression data",
                        type=str)
    parser.add_argument("drug_file",
                        help="The path the csv file containing preprocessed drug treatment data",
                        type=str)
    args = vars(parser.parse_args())

    exprs_path_in = args["expression_file"]
    exprs_path_out = re.sub(r"\.tsv", ".csv", exprs_path_in)
    clin_path = args["drug_file"]

    print("Loading gene expression data", end="... ", flush=True)
    exprs_data = pd.read_csv(exprs_path_in, index_col=0, sep="\t")
    print("Done", flush=True)

    print("Loading preprocessed drug data", end="... ", flush=True)
    clin_data = pd.read_csv(clin_path, index_col="aliquot_submitter_id")
    print("Done", flush=True)

    print("Cleaning up data", end="... ", flush=True)
    exprs_data = exprs_data.transpose()
    clin_data = clin_data[clin_data.index.isin(exprs_data.index)]  # Use clinical data that have expression data
    exprs_data = exprs_data[exprs_data.index.isin(clin_data.index)]  # Use expression data that have clinical data
    exprs_data = exprs_data.loc[clin_data.index, :]  # Make sure the order of the rows match
    # There may be biopsies with NaN values for certain genes; these will be dealt with later
    exprs_data = np.log1p(exprs_data)  # Log-transform the expression data
    print("Done", flush=True)

    # Save mappings of feature names to KEGG hsa IDs, ENTREZ IDs, and gene symbols
    print("Creating a mapping of feature names to KEGG hsa IDs, ENTREZ IDs, and gene symbols", end="... ", flush=True)
    orig_feat_names = exprs_data.columns
    reg_exp1 = re.compile(r"^[^|]+")  # Match gene symbol
    reg_exp2 = re.compile(r"(?<=\|)\d+")  # Match ENTREZ ID
    gene_symbols = [reg_exp1.search(string).group(0) for string in orig_feat_names]
    entrez_ids = [reg_exp2.search(string).group(0) for string in orig_feat_names]
    kegg_ids = [f"hsa{entrez}" for entrez in entrez_ids]
    feature_map = pd.DataFrame({
        "original": orig_feat_names,
        "entrez": entrez_ids,
        "symbol": gene_symbols,
        "kegg": kegg_ids
    })
    feature_map.to_csv("data/feature_map.csv", index_label=False)
    new_names = dict(zip(orig_feat_names, kegg_ids))
    exprs_data.rename(columns=new_names, inplace=True)  # Replace original names with KEGG IDs
    print("Done", flush=True)

    print("Writing data to disk", end="... ", flush=True)
    exprs_data.to_csv(exprs_path_out)
    clin_data.to_csv(clin_path, index_label="aliquot_submitter_id")
    print("Done", flush=True)

    # Remove the gene expression tsv file
    os.unlink(exprs_path_in)


if __name__ == "__main__":
    main()
