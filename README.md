# GNNChemo
This repository contains scripts that train graph neural networks and sparse MLPs to predict responses to 
chemotherapy recorded in the Cancer Genome Atlas (TCGA) datasets. The GNNs are applied to Reactome biological pathway
graphs. Each node represents a gene; its associated feature is biopsy gene expression measured by RNA sequencing assays.

# Setup
Clone GitHub repositories. The `gnnchemo` repository contains the project files. The `PathGNN` repository contains 
additional data files that are used to create the pathway graphs.
```bash
ROOT_DIR=$(pwd)
git clone git@github.com:mdparkes/gnnchemo.git
git clone git@github.com:BioAI-kits/PathGNN.git
unzip $ROOT_DIR/PathGNN/Pathway/pathways.zip -d $ROOT_DIR/PathGNN/Pathway
cp -r $ROOT_DIR/PathGNN/Pathway $ROOT_DIR/gnnchemo/data
```
Set up a virtual environment, e.g. with conda and activate it, and enter the project directory.
```bash
conda create -n gnnchemo python==3.11.7
conda activate gnnchemo
cd $ROOT_DIR/gnnchemo
```
Install `keggpathwaygraphs` and other required libraries in the virtual environment:
```bash
pip install git+https://github.com/mdparkes/keggpathwaygraphs.git
pip install -r $ROOT_DIR/gnnchemo/requirements.txt
```
# Data preparation

Download TCGA gene expression data to the data directory and clean the data:
```bash
FILE_URL=https://api.gdc.cancer.gov/data/3586c0da-64d0-4b74-a449-5ff4d9136611
curl $FILE_URL -o $ROOT_DIR/gnnchemo/data/tcga_exprs.tsv
python3 src/process_data.py data/tcga_exprs.tsv data/processed_drug_df.csv
```
Scrape BRITE orthology data and create graph information dictionaries:
```bash
python3 -m keggpathwaygraphs.create_graph --output_dir $ROOT_DIR/gnnchemo/data
```
Create Reactome graph information objects and PyTorch graph Data objects. This will also create training, validation,
and test partitions of the dataset.
```bash
python3 src/create_reactome_graph.py --data_dir data --pathway_dir data/Pathway
python3 src/create_pyg_graph_objs.py \
  --exprs_file data/tcga_exprs.csv \
  --drug_file data/processed_drug_df.csv \
  --output_dir data
```
Create Data objects for input to MLP models:
```bash
python3 src/create_mlp_inputs.py \
  --exprs_file data/tcga_exprs.csv \
  --drug_file data/processed_drug_df.csv \
  --output_dir data
```
# Execute an experiment
Tune hyperparameters for the GNN and MLP models. Be aware that the GNN models may take days to train.
```bash
python3 src/tune_gnn_hyperparameters.py \
  --data_dir data \
  --output_dir test_experiment \
  --batch_size 48 \
  --use_drug_input
  
python3 src/tune_mlp_hyperparameters.py \
  --data_dir data \
  --output_dir test_experiment \
  --batch_size 48 \
  --use_drug_input
```
Train models with and without drug information as auxiliary inputs. Models will train in parallel over multiple CPU 
cores. The `--num_workers` argument is the number of CPU cores used for training. The number should evenly divide 
the batch size, and there must be adequate RAM to use the specified number of workers. The training examples in a 
minibatch are divided among `--num_workers` CPU cores. For example, if `--batch_size` is 48 and `--num_workers` is 4,
each worker handles 12 examples from the minibatch.
```bash
python3 src/train_gnn_models.py \
  --data_dir data \
  --output_dir test_experiment \
  --batch_size 48 \
  --num_workers 4 \
  --use_drug_input

python3 src/train_gnn_models.py \
  --data_dir data \
  --output_dir test_experiment \
  --batch_size 48 \
  --num_workers 4

python3 src/train_mlp_models.py \
  --data_dir data \
  --output_dir test_experiment \
  --batch_size 48 \
  --num_workers 4 \
  --use_drug_input
  
python3 src/train_mlp_models.py \
  --data_dir data \
  --output_dir test_experiment \
  --batch_size 48 \
  --num_workers 4
```
Export results from all models as csv files:
```bash
python3 src/export_gnn_experiment_data.py \
  --data_dir data \
  --output_dir test_experiment \
  --batch_size 48 \
  --use_drug_input

python3 src/export_gnn_experiment_data.py \
  --data_dir data \
  --output_dir test_experiment \
  --batch_size 48

python3 src/export_mlp_experiment_data.py \
  --data_dir data \
  --output_dir test_experiment \
  --batch_size 48 \
  --use_drug_input
  
python3 src/export_mlp_experiment_data.py \
  --data_dir data \
  --output_dir test_experiment \
  --batch_size 48
```

# Notes
The file `data/processed_drug_df.csv` contains the fully preprocessed drug data for modeling. The files 
`src/get_gdc_data.R` and `src/process_drug_df.R` are only included for reference and should not be run. Manual data 
processing steps were performed between scraping the data with `src/get_gdc_data.R` and writing the final 
preprocessed file with `src/process_drug_df.R`


## Troubleshooting

If conda is slow to solve the environment, try setting the channel priority to strict:

```commandline
conda config --set channel_priority strict
```
