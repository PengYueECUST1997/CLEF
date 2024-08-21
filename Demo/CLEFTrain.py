from Demo_utils import train_clef
from argparse import ArgumentParser



def main(args):
    input_file_config = {'fasta':args.Fa, "supp_feat":args.Feat}
    train_config = {'lr':args.lr, 'batch_size':args.btz, 'num_epoch':args.epoch}
    output_dir = args.Out
    esm_config = {'pretrained_model_params':esm_model_path} if args.esm_model_path else None
    config = {
      'input_file_config':input_file_config,
      'output_dir':output_dir,
      'train_config':train_config,
      'esm_config':esm_config
    }
    train_clef(**config)
    
    
    
    
    




if __name__ == '__main__':
    parser = ArgumentParser(description="Train CLEF with protein sequence [fasta file] and supplemental features [arrays].")
    parser.add_argument('--Fa', required=True,  type=str,
                        help="fasta file of input proteins")
    parser.add_argument('--Feat', required=True,  type=str,
                        help="array file of input features")
    parser.add_argument('--Out', required=True,  type=str,
                        help="output directory of ")
    parser.add_argument('--esm_model_path', default=None, type=str,
                        help="Local ESM pretrained model path. (default: None, using downloaded ESM2-650M from fair-esm)")
    parser.add_argument('--lr', default=0.0002,  type=float,
                        help="Learning rate for training")
    parser.add_argument('--btz', default=128,  type=int,
                        help="Batch size for training")
    parser.add_argument('--epoch', default=20,  type=int,
                        help="Number of epoch for training")
    args = parser.parse_args()
    
    main(args)

