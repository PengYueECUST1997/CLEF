from Demo_utils import generate_protein_representation
from argparse import ArgumentParser
import os
from pathlib import Path


def main(args):
    mode = args.mode
    esm_model_path = args.esm_model_path
    model_params_path = args.weight
    input_file = args.In
    output_file = args.Out
    maxlength = args.Maxlen
    esm_config = {'pretrained_model_params':esm_model_path} if esm_model_path else None
    
    model_params_path = Path(os.path.abspath(model_params_path))
    config = {
      'input_file':input_file,
      'output_file':output_file,
      'model_params_path':model_params_path,
      'esm_config':esm_config,
      'maxlength':256,
      'mode':mode
        
    }
    generate_protein_representation(**config)




if __name__ == '__main__':
    parser = ArgumentParser(description="Transform protein sequence [fasta file] into encoded representation [arrays].")
    
    parser.add_argument('--In', required=True,  type=str,
                        help="fasta file of input proteins")
    parser.add_argument('--Out', required=True,  type=str,
                        help="output protein representation arrays path")
    parser.add_argument('--Maxlen', default=256,  type=int,
                        help="Max length of protein sequence input into model")
    parser.add_argument('--mode', default='clef', choices=['esm', 'clef'], type=str,
                        help="Using clef-generated cross-modality representations or direct using ESM2 embeddings. [clef , esm]")
    parser.add_argument('--esm_model_path', default=None, type=str,
                        help="Local ESM pretrained model path. (default: None, using downloaded ESM2-650M from fair-esm)")
    parser.add_argument('--weight', required=True,  type=str,
                        help="CLEF model weights path.")
    args = parser.parse_args()
    
    main(args)
