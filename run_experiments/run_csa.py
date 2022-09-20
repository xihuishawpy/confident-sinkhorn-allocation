import sys
import os

sys.path.insert(0,'..')
#sys.path.append('..')
sys.path.append("../algorithm")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import os
import argparse
import logging
import pickle
from tqdm import tqdm
from algorithm.pseudo_labeling import Pseudo_Labeling
#from confident_sinkhorn_allocation.algorithm.flexmatch import FlexMatch
#from confident_sinkhorn_allocation.algorithm.ups import UPS
from algorithm.csa import CSA


from utilities.utils import get_train_test_unlabeled,append_acc_early_termination
from utilities.utils import get_train_test_unlabeled_for_multilabel

import warnings
warnings.filterwarnings('ignore')


def run_experiments(args, save_dir):

    out_file = args.output_filename
    numTrials=args.numTrials
    numIters=args.numIters
    confidence_choice = args.confidence_choice
    verbose=args.verbose
    dataset_name=args.dataset_name
    num_XGB_models=args.numXGBs


    IsMultiLabel = dataset_name in ['yeast','emotions']
    accuracy = []


    for tt in tqdm(range(numTrials)):
        
        np.random.seed(tt)

        # load the data
        x_train, y_train, x_test, y_test, x_unlabeled = (
            get_train_test_unlabeled_for_multilabel(
                dataset_name,
                path_to_data='../all_data_multilabel.pickle',
                random_state=tt,
            )
            if IsMultiLabel
            else get_train_test_unlabeled(
                dataset_name,
                path_to_data='../all_data.pickle',
                random_state=tt,
            )
        )

        pseudo_labeller = CSA(x_unlabeled,x_test,y_test, 
                num_iters=numIters,
                confidence_choice=confidence_choice,
                num_XGB_models=num_XGB_models,
                verbose = verbose,
                IsMultiLabel=IsMultiLabel
            )
        pseudo_labeller.fit(x_train, y_train)

        accuracy.append(  append_acc_early_termination(pseudo_labeller.test_acc,numIters) )


    # print and pickle results
    filename = os.path.join(
        save_dir,
        f'{out_file}_{pseudo_labeller.algorithm_name}_{dataset_name}_M_{num_XGB_models}_numIters_{numIters}_numTrials_{numTrials}.pkl',
    )

    print('\n* Trial summary: avgerage of accuracy per Pseudo iterations')
    print( np.mean( np.asarray(accuracy),axis=0))
    print(f'\n* Saving to file {filename}')
    with open(filename, 'wb') as f:
        pickle.dump([accuracy], f)
        f.close()

def main(args):

    # make save directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = f'{save_dir}//'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    run_experiments(args, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for CSA experiments')
    parser.add_argument('--numIters', type=int, default=5, help='number of Pseudo Iterations')
    parser.add_argument('--numTrials', type=int, default=20, help ='number of Trials (Repeated Experiments)' )
    parser.add_argument('--numXGBs', type=int, default=10, help ='number of XGB models, M=?' )
    parser.add_argument('--confidence_choice', type=str, default='ttest', help ='confidence choices: ttest | variance | entropy | None' )
    parser.add_argument('--dataset_name', type=str, default='synthetic_control_6c', help='segment_2310_20 | wdbc_569_31 | analcatdata_authorship | synthetic_control_6c | \
        German-credit |  madelon_no | agaricus-lepiota | breast_cancer | digits | yeast | emotions')
    parser.add_argument('--verbose', type=str, default='True', help='verbose True or False')
    parser.add_argument('--output_filename', type=str, default='', help='name of output files')
    parser.add_argument('--save_dir', type=str, default='results_output', help='name of save directory')

    args = parser.parse_args()
    main(args)