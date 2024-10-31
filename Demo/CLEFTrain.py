from Demo_utils import train_clef
from argparse import ArgumentParser



def main(args):
    seq_path = args.Seq
    modal_paths = args.Feat
    modal_path_dict = {f'modal_{i}':path for i,path in enumerate(modal_paths)}
    input_file_config = {'seq':seq_path}
    input_file_config.update(modal_path_dict)
    train_config = {'lr':args.lr, 'batch_size':args.btz, 'num_epoch':args.epoch, 'maxlen':args.maxlen}
    output_dir = args.Out
    esm_config = {'maxlen':args.maxlen}
    if args.esm_model_path:
        esm_config['pretrained_model_params'] = args.esm_model_path
    config = {
      'input_file_config':input_file_config,
      'output_dir':output_dir,
      'train_config':train_config,
      'esm_config':esm_config
    }
    # print(input_file_config, output_dir, train_config, esm_config)
    train_clef(**config)
    
    
    
    
    




if __name__ == '__main__':
    parser = ArgumentParser(description="Train CLEF with protein sequence [fasta file] and supplemental features [arrays].")
    parser.add_argument('--Seq', required=True,  type=str,
                        help="Sequence file of input proteins, can be a fasta text or 2D representations")
    parser.add_argument('--Feat', nargs='+', required=True,  type=str,
                        help="array file of input features")
    parser.add_argument('--Out', required=True,  type=str,
                        help="output directory of ")
    parser.add_argument('--esm_model_path', default=None, type=str,
                        help="Local ESM pretrained model path. (default: None, using downloaded ESM2-650M from fair-esm)")
    parser.add_argument('--lr', default=0.00002,  type=float,
                        help="Learning rate for training")
    parser.add_argument('--btz', default=128,  type=int,
                        help="Batch size for training")
    parser.add_argument('--epoch', default=20,  type=int,
                        help="Number of epoch for training")
    parser.add_argument('--maxlen', default=256,  type=int,
                        help="Max_length of protein input into model")
    args = parser.parse_args()
    
    main(args)

