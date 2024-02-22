# NOTE
This repository is a work in progress. The scripts within are not complete.


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
Download TCGA gene expression data to the data directory:
```bash
FILE_URL=http://api.gdc.cancer.gov/data/3586c0da-64d0-4b74-a449-5ff4d9136611
curl -o $ROOT_DIR/gnnchemo/data/tcga_exprs.tsv $FILE_URL
```

## Troubleshooting

If conda is slow to solve the environment, try setting the channel priority to strict:

```commandline
conda config --set channel_priority strict
```
