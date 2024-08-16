# CLEF
Contrastive Learning of language Embedding and biological Feature is a contrastive learning framework used to combine information from supplemental biological features or experimental data with protein language model [ESM2](https://github.com/facebookresearch/esm) generated embeddings. Generated cross-modal feature can be used for better downstream protein prediction like Gram-negative bacterial effectors prediction or virulence factors discovery.


![](./Material/Main.jpg)

## Set up

### Requirement
The project is implemented with python (3.11.3), the following library packages are required:

```txt
torch==2.0.1
fair-esm==2.0.0
biopython==1.79
einops==0.7.0
numpy==1.23.4
scikit-learn==1.1.3
```
We also tested the code with recent versions of these requirements, other reasonable versions should work as well.
The code was tested on Windows.

### Installation

Required Python packages are listed in `requirements.txt` file.

To install the required packages, run the following command using pip:
```shell
pip install -r requirements.txt
```

To output the result table, the package `pandas` was used in code: 
```shell
pip install pandas 
```

## Demo

CLEF was trained under a contrative learning framework, and can generate cross-modal representations based on pre-trained protein language models (pLMs) of [ESM2](https://github.com/facebookresearch/esm)
The generated cross-modal representations can be used in other downstream predictions task and enhance the protein classification performance.

### Generate Cross-Modal Representation

We provide some example code in `Demo` to use the CLEF, you can generate  cross-modal representations based on ESM2:
```shell
cd ./Demo
python .\GenerateCrossModalityRep.py --In .\Test_demo.faa --Out .\Test_demo_clef --weight ..\pretrained_model\Demo_clef_dpc_pssm_encoder.pt --supp_feat_dim 400 
```
 Parameters:

- `--In` fasta file of input proteins.
- `--Out` output protein representation arrays file.
- `--weight` pretrained CLEF model parameters path, here we use the example model `Demo_clef_dpc_pssm_encoder.pt` trained by DPC-PSSM feature in  `pretrained_model`
- `--supp_feat_dim` numbers of dimensions of biological features used for CLEF (need to match with pretrained CLEF model parameters)

This will create a file `Test_demo_clef`, containing the cross-modal representations of input proteins

 ### Predict using the Protein Representation

After that, the generated representations can be used in other protein classification tasks, here we give an example of T6SE prediction:

```shell
python .\PredictProteinClassification.py --In .\Test_demo_clef --Out Test_result.xlsx --weight ..\pretrained_model\clef_dpcpssm_T6_classifier.pt
```
Parameters:

- `--In` fasta file of input proteins.
- `--Out` output protein representation arrays file.
- `--weight` classifier weights path, `clef_dpcpssm_T6_classifier.pt` is a simple multilayer perceptron (MLP) trained to discriminate T6SE and non-T6SE



## Contact

Please contact Yue Peng at [756028108@qq.com] for questions.

