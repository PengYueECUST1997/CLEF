import pandas as pd
from Demo_utils import predict_from_1D_rep
from argparse import ArgumentParser
from Module import test_dnn

def main(args):

    input_file = args.In 
    output_file = args.Out
    cutoff = args.cutoff
    classifier_path = args.weight   
    classifer = test_dnn
    config = {
      'input_file':input_file,
      'output_file':output_file,
      'initial_model':classifer,
      'params_path':classifier_path,
      'cutoff':cutoff,
      'Return':True
    }        
    output_dict = predict_from_1D_rep(**config)
    print(output_dict.iloc[:10,:])
    print(f"Predictions saved at {output_file}")
    

if __name__ == '__main__':
    parser = ArgumentParser(description="Predict binary class using protein representations [arrays]")
    
    parser.add_argument('--In', required=True,  type=str,
                        help="arrays file of input proteins")
    parser.add_argument('--Out', required=True,  type=str,
                        help="output result [csv table]")
    parser.add_argument('--weight', required=True,  type=str,
                        help="Classifier weights path.")
    parser.add_argument('--cutoff', required=False,  type=float,
                        help="Cutoff used to predict binary class label")

    args = parser.parse_args()
    
    main(args)
