# NOTE
This repository is a work in progress. The scripts within are not complete.


# Setup
Set up a virtual environment and activate it. Set the working directory to gnnchemo.

Clone `PathGNN`, unzip the contents of `PathGNN/Pathway/pathways.zip`, and copy PathGNN/Pathway to the gnnchemo 
data directory:

```commandline
git clone git@github.com:BioAI-kits/PathGNN.git
cd /path/to/PathGNN/Pathway
unzip pathways.zip
cp -r /path/to/PathGNN/Pathway /path/to/gnnchemo/data
```

Install `keggpathwaygraphs`

```command line
python3 -m pip install git+https://github.com/mdparkes/keggpathwaygraphs.git
```

## Troubleshooting

If conda is slow to solve the environment, try setting the channel priority to strict:

```commandline
conda config --set channel_priority strict
```
