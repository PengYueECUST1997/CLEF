import os
import pandas as pd
import sys


    
from Demo_utils import predict_from_1D_rep
from argparse import ArgumentParser
from Module import test_dnn

def main(args):

    input_file = args.In 
    output_file = args.Out
    cutoff = args.cutoff
    rep_path = input_file
    classifier_config = {'num_embeds':1280, "out_dim":1}
    classifier_path = args.weight   
    classifer = test_dnn(**classifier_config)
    config = {
      'rep_path':rep_path,
      'output_file':None,
      'model':classifer,
      'params_path':classifier_path,
      'Return':True
    }        
    output_dict = predict_from_1D_rep(**config)
    output_dict.columns = [x if i <len(output_dict.columns) - 1 else 'score' for i,x in enumerate(output_dict.columns)]
    output_dict['Predict'] = ['Yes' if x >= cutoff else 'No' for x in output_dict['score']]
    output_dict.to_excel(output_file, index = False)
    print(output_dict.iloc[:10,:])
    print(f"Predictions saved at {output_file}")
    

if __name__ == '__main__':
    parser = ArgumentParser(description="Predict binary class using protein representations [arrays]")
    
    parser.add_argument('--In', required=True,  type=str,
                        help="arrays file of input proteins")
    parser.add_argument('--Out', required=True,  type=str,
                        help="output result [xlsx table]")
    parser.add_argument('--weight', required=True,  type=str,
                        help="Classifier weights path.")
    parser.add_argument('--cutoff', default=0.5,  type=float,
                        help="Cutoff used to predict binary class label")

    args = parser.parse_args()
    
    main(args)
