import os
import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #specify GPU to use
from run_nn_models import run_nn_models
from utils import unseen_modality_test, diff_specs, ntrain_combine_df, frac_combine_df


t_start = time.time()
##################USER-DEFINED PARAMETERS##################
# Where data will be saved: rootpath + dataset + '/'
rootpath = '/home/eduard/git/HTNet/'
dataset = 'move_rest_ecog'

# Data load paths
ecog_lp = rootpath + 'ecog_data/' # data load path
ecog_roi_proj_lp = rootpath + 'proj-matrices-ecog/'

### Same modality decoder params (across participants) ###
n_folds_same = 2#36 # number of total folds
spec_meas_same = ['power']
hyps_same = {'F1' : 19, 'dropoutRate' : 0.5, 'kernLength' : 24,
             'kernLength_sep' : 88, 'dropoutType' : 'Dropout',
             'D' : 2, 'n_estimators' : 240, 'max_depth' : 6}
hyps_same['F2'] = hyps_same['F1'] * hyps_same['D'] # F2 = F1 * D
epochs_same = 100
patience_same = 15

### Fine-tune same modality decoders ###
model_type_finetune = 'eegnet_hilb' # NN model type to fine-tune (must be either 'eegnet_hilb' or 'eegnet')
layers_to_finetune = ['all',['conv2d','batch_normalization'],
                      ['batch_normalization','depthwise_conv2d','batch_normalization_1'],
                      ['separable_conv2d','batch_normalization_2']]
# Options:  'all' - allow entire model to be retrained
#           ['conv2d','batch_normalization']
#           ['batch_normalization','depthwise_conv2d','batch_normalization_1']
#           ['separable_conv2d','batch_normalization_2']
#           None - transfer learning of new last 3 layers
sp_finetune = [rootpath + dataset + '/tf_all_per/',
               rootpath + dataset + '/tf_per_1dconv/',
               rootpath + dataset + '/tf_depth_per/',
               rootpath + dataset + '/tf_sep_per/'] # where to save output (should match layers_to_finetune)

# How much train/val data to use, either by number of trials or percentage of available data
use_per_vals = True #if True, use percentage values (otherwise, use number of trials)
per_train_trials = [.17,.33,.5,0.67]
per_val_trials = [.08,.17,.25,0.33]
n_train_trials = [16,34,66,100]
n_val_trials = [8,16,34,50]

### Train same modality decoders with different numbers of training participants ###
max_train_parts = 10 #'rnn' use 1--max_train_subs training participants
n_val_parts = 1 # number of validation participants to use
##################USER-DEFINED PARAMETERS##################

#### Same modality training ####
for s,val in enumerate(spec_meas_same):
    do_log = True if val == 'power_log' else False
    compute_val = 'power' if val == 'power_log' else val
    multi_sp = rootpath + dataset  + '/combined_sbjs_' + val + '/'
    if not os.path.exists(multi_sp):
        os.makedirs(multi_sp)
    combined_sbjs = True
    models = ['rnn']#, 's4', 'ncde'] # 'rnn', 's4', 'ncde' avoid fitting non-HTNet models again
    accuracy = run_nn_models(multi_sp, n_folds_same, combined_sbjs, ecog_lp, ecog_roi_proj_lp, test_day = 'last', do_log=do_log,
                  epochs=epochs_same, patience=patience_same, models=models, compute_val=compute_val,
                  F1 = hyps_same['F1'], dropoutRate = hyps_same['dropoutRate'], kernLength = hyps_same['kernLength'],
                  kernLength_sep = hyps_same['kernLength_sep'], dropoutType = hyps_same['dropoutType'],
                  D = hyps_same['D'], F2 = hyps_same['F2'], n_estimators = hyps_same['n_estimators'], max_depth = hyps_same['max_depth'])
    print(accuracy)

# try:
#     ntrain_combine_df(rootpath + dataset)
#     frac_combine_df(rootpath + dataset, ecog_roi_proj_lp)
# except BaseException:
#     pass

#### Pre-compute difference spectrograms for ECoG and EEG datasets ####
#diff_specs(rootpath + dataset + '/combined_sbjs_power/', ecog_lp, ecog=True)

print('Elapsed time: ' + str(time.time() - t_start))
