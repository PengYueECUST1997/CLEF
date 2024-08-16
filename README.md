# CLEF
Contrastive Learning of language Embedding and biological Feature is a contrastive learning framework used to combine information from supplemental biological features or experimental data with protein language model [ESM-2](https://github.com/facebookresearch/esm) generated embeddings. Generated cross-modal feature can be used for better downstream protein prediction like Gram-negative bacterial effectors prediction or virulence factors discovery.


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



## Contact

Please contact Yue Peng at [756028108@qq.com] for questions.

