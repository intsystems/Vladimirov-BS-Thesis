import os
import pickle
import time
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

from htnet import HTNet, HTNetWithRNN, HTNetWithNCDE, HTNetWithS4
from utils import load_data, folds_choose_subjects, subject_data_inds, roi_proj_rf, \
    get_custom_motor_rois, proj_mats_good_rois, to_categorical

if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    # Choose GPU 0 as a default if not specified (can set this in Python script that calls this)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 20


def unroll_batch(batch, projectROIs):
    if projectROIs:
        inputs, rois, labels = batch
        inputs = inputs.to(DEVICE)
        rois = rois.to(DEVICE)
        labels = labels.to(DEVICE)
    else:
        inputs, labels = batch
        inputs = inputs.to(DEVICE)
        rois = None
        labels = labels.to(DEVICE)

    return inputs, rois, labels


def cnn_model(X_train, Y_train, X_validate, Y_validate, X_test, Y_test, chckpt_path, modeltype,
              proj_mat_out=None, sbj_order_train=None, sbj_order_validate=None,
              sbj_order_test=None, nROIs=100, nb_classes=2, dropoutRate=0.25,
              kernLength=32, F1=8, D=2, F2=16, dropoutType='Dropout',
              kernLength_sep=16,
              patience=5, do_log=False, epochs=20,
              compute_val='power', ecog_srate=500):
    '''
    Perform NN model fitting based on specified parameters.
    '''
    # Logic to determine how to run model
    projectROIs = True if proj_mat_out is not None else False  # True if there are multiple subjects in train data
    useHilbert = True if modeltype == 'conv' else False  # True if want to use Hilbert transform layer

    # Load NN model
    if modeltype == 'conv':
        model = HTNet(nb_classes, Chans=X_train.shape[1], Samples=X_train.shape[-1],
                      dropoutRate=dropoutRate, kernLength=kernLength, F1=F1, D=D, F2=F2,
                      dropoutType=dropoutType, kernLength_sep=kernLength_sep,
                      ROIs=nROIs, useHilbert=useHilbert, projectROIs=projectROIs, do_log=do_log,
                      compute_val=compute_val, data_srate=ecog_srate).to(DEVICE)
    elif modeltype == 'rnn':
        model = HTNetWithRNN(nb_classes, Chans=X_train.shape[1], Samples=X_train.shape[-1],
                      dropoutRate=dropoutRate, kernLength=kernLength, F1=F1, D=D, F2=F2,
                      dropoutType=dropoutType, kernLength_sep=kernLength_sep,
                      ROIs=nROIs, useHilbert=useHilbert, projectROIs=projectROIs, do_log=do_log,
                      compute_val=compute_val, data_srate=ecog_srate, k_signals=X_train.shape[2]).to(DEVICE)
    elif modeltype == 'ncde':
        model = HTNetWithNCDE(nb_classes, Chans=X_train.shape[1], Samples=X_train.shape[-1],
                      dropoutRate=dropoutRate, kernLength=kernLength, F1=F1, D=D, F2=F2,
                      dropoutType=dropoutType, kernLength_sep=kernLength_sep,
                      ROIs=nROIs, useHilbert=useHilbert, projectROIs=projectROIs, do_log=do_log,
                      compute_val=compute_val, data_srate=ecog_srate, k_signals=X_train.shape[2]).to(DEVICE)
    elif modeltype == 's4':
        model = HTNetWithS4(nb_classes, Chans=X_train.shape[1], Samples=X_train.shape[-1],
                              dropoutRate=dropoutRate, kernLength=kernLength, F1=F1, D=D, F2=F2,
                              dropoutType=dropoutType, kernLength_sep=kernLength_sep,
                              ROIs=nROIs, useHilbert=useHilbert, projectROIs=projectROIs, do_log=do_log,
                              compute_val=compute_val, data_srate=ecog_srate, k_signals=X_train.shape[2]).to(DEVICE)
    else:
        raise ValueError('Wrong modeltype!')

    # Set up optimizer, checkpointer, and early stopping during model fitting
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    # stop if val_loss doesn't improve after certain # of epochs

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    X_validate_tensor = torch.tensor(X_validate, dtype=torch.float32)
    Y_validate_tensor = torch.tensor(Y_validate, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    if proj_mat_out is not None:
        proj_mat_out = torch.tensor(proj_mat_out, dtype=torch.float32)

    # Create PyTorch datasets
    if projectROIs:
        train_dataset = TensorDataset(X_train_tensor, proj_mat_out[sbj_order_train], Y_train_tensor)
        validate_dataset = TensorDataset(X_validate_tensor, proj_mat_out[sbj_order_validate, ...], Y_validate_tensor)
        test_dataset = TensorDataset(X_test_tensor, proj_mat_out[sbj_order_test, ...], Y_test_tensor)
    else:
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        validate_dataset = TensorDataset(X_validate_tensor, Y_validate_tensor)
        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    # Create PyTorch data loaders
    print('Train:', len(train_dataset), 'Validation:', len(validate_dataset), 'test:', len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Perform model fitting in PyTorch
    best_val_loss = float('inf')
    best_epoch = 0
    last_epoch = epochs - 1
    t_start_fit = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, rois, labels = unroll_batch(batch, projectROIs)

            optimizer.zero_grad()
            outputs = model(inputs, rois)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in validate_loader:
                inputs, rois, labels = unroll_batch(batch, projectROIs)

                outputs = model(inputs, rois)
                loss = loss_func(outputs, labels)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(validate_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

        # Save the best model
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            torch.save(model.state_dict(), chckpt_path)
            torch.save(optimizer.state_dict(), chckpt_path + '-best-optimizer')

        # Early stopping
        if epoch - best_epoch >= patience:
            last_epoch = epoch
            print("Early stopping!")
            break

    t_fit_total = time.time() - t_start_fit

    # Load model weights from best model and compute train/val/test accuracies
    model.load_state_dict(torch.load(chckpt_path))
    model.eval()

    accs_lst = []
    with torch.no_grad():
        accuracy = []
        for batch in train_loader:
            inputs, rois, labels = unroll_batch(batch, projectROIs)
            batch_preds = model(inputs, rois).argmax(axis=-1)
            accuracy.extend(batch_preds.cpu() == labels.cpu())

        accs_lst.append(np.mean(accuracy))

        accuracy = []
        for batch in validate_loader:
            inputs, rois, labels = unroll_batch(batch, projectROIs)
            batch_preds = model(inputs, rois).argmax(axis=-1)
            accuracy.extend(batch_preds.cpu() == labels.cpu())

        accs_lst.append(np.mean(accuracy))

        accuracy = []
        for batch in test_loader:
            inputs, rois, labels = unroll_batch(batch, projectROIs)
            batch_preds = model(inputs, rois).argmax(axis=-1)
            accuracy.extend(batch_preds.cpu() == labels.cpu())

        accs_lst.append(np.mean(accuracy))

    return accs_lst, torch.tensor([last_epoch, t_fit_total])


def run_nn_models(sp, n_folds, combined_sbjs, lp, roi_proj_loadpath,
                  pats_ids_in=['EC01', 'EC02', 'EC03', 'EC04', 'EC05', 'EC06',
                               'EC07', 'EC08', 'EC09', 'EC10', 'EC11', 'EC12'],
                  n_evs_per_sbj=500, test_day=None, tlim=[-1, 1],
                  n_chans_all=140, dipole_dens_thresh=0.2, rem_bad_chans=True,
                  models=['eegnet_hilb'], save_suffix='',
                  n_estimators=150, max_depth=8, overwrite=True, dropoutRate=0.25, kernLength=32,
                  F1=8, D=2, F2=16, dropoutType='Dropout', kernLength_sep=16, rand_seed=1337,
                  patience=5, do_log=False, n_test=1, n_val=4,
                  custom_rois=True, n_train=7, epochs=20, compute_val='power', ecog_srate=500,
                  half_n_evs_test='nopad', trim_n_chans=True):
    '''
    Main function that prepares data and aggregates accuracy values from model fitting.
    Note that overwrite variable no longer does anything.
    Also note that ecog_srate is only needed for frequency sliding computation in neural net (if compute_val=='freqslide')
    '''
    # Ensure pats_ids_in and models variables are lists
    if not isinstance(pats_ids_in, list):
        pats_ids_in = [pats_ids_in]
    if not isinstance(models, list):
        models = [models]

    # Save pickle file with dictionary of input parameters (useful for reproducible dataset splits and model fitting)
    params_dict = {'sp': sp, 'n_folds': n_folds, 'combined_sbjs': combined_sbjs, 'lp': lp, 'pats_ids_in': pats_ids_in,
                   'n_evs_per_sbj': n_evs_per_sbj, 'test_day': test_day, 'tlim': tlim, 'n_chans_all': n_chans_all,
                   'dipole_dens_thresh': dipole_dens_thresh, 'rem_bad_chans': rem_bad_chans, 'models': models,
                   'save_suffix': save_suffix, 'n_estimators': n_estimators, 'max_depth': max_depth,
                   'overwrite': overwrite,
                   'dropoutRate': dropoutRate, 'kernLength': kernLength, 'F1': F1, 'D': D, 'F2': F2,
                   'dropoutType': dropoutType,
                   'kernLength_sep': kernLength_sep, 'rand_seed': rand_seed,
                   'patience': patience, 'do_log': do_log, 'n_test': n_test,
                   'n_val': n_val, 'n_train': n_train, 'epochs': epochs, 'compute_val': compute_val,
                   'ecog_srate': ecog_srate, 'trim_n_chans': trim_n_chans}
    f = open(sp + 'param_file.pkl', 'wb')
    pickle.dump(params_dict, f)
    f.close()

    # Set random seed
    np.random.seed(rand_seed)

    # Perform different procedures depending on whether or not multiple subjects are being fit together
    # For multi-subject fits, obtain projection matrix and good regions of interest for each subject
    if custom_rois:
        custom_roi_inds = get_custom_motor_rois()  # load custom roi's from precentral, postcentral, and inf parietal (AAL2)
    else:
        custom_roi_inds = None
    print("Determining ROIs")
    proj_mat_out, good_ROIs, chan_ind_vals_all = proj_mats_good_rois(pats_ids_in,
                                                                     n_chans_all=n_chans_all,
                                                                     rem_bad_chans=rem_bad_chans,
                                                                     dipole_dens_thresh=dipole_dens_thresh,
                                                                     custom_roi_inds=custom_roi_inds,
                                                                     chan_cut_thres=n_chans_all,
                                                                     roi_proj_loadpath=roi_proj_loadpath)
    nROIs = len(good_ROIs)
    print("ROIs found")

    # Retain only the electrodes with nonzero data (initially padded because number of electrodes varies across subjects)
    # proj_mat_out : (len(pats_ids_in) x len(good_ROIs) x n_chans_all)
    if trim_n_chans:
        n_chans_all = len(np.nonzero(proj_mat_out.reshape(-1, proj_mat_out.shape[-1]).mean(axis=0))[0])
        proj_mat_out = proj_mat_out[..., :n_chans_all]
    np.save(sp + "proj_mat_out", proj_mat_out)

    # Load ECoG data (if test_day is None, then X_test_orig, y_test_orig, and sbj_order_test_load will be empty)
    X, y, X_test_orig, y_test_orig, sbj_order, sbj_order_test_load = load_data(pats_ids_in, lp,
                                                                               n_chans_all=n_chans_all,
                                                                               test_day=test_day, tlim=tlim)
    X[np.isnan(X)] = 0  # set all NaN's to 0
    y = y.astype(int)
    y_test_orig = y_test_orig.astype(int)
    # Identify the number of unique labels (or classes) present
    nb_classes = len(np.unique(y))

    # Choose which subjects for training/validation/testing for every fold (splits are based on random seed)
    sbj_inds_all_train, sbj_inds_all_val, sbj_inds_all_test = folds_choose_subjects(n_folds, pats_ids_in,
                                                                                    n_test=n_test, n_val=n_val,
                                                                                    n_train=n_train)

    # Iterate across all model types specified
    labels_unique = np.unique(y)
    if isinstance(n_evs_per_sbj, str):
        half_n_evs = n_evs_per_sbj
    else:
        half_n_evs = n_evs_per_sbj // len(labels_unique)

    train_inds_folds, val_inds_folds, test_inds_folds = [], [], []
    result = []

    for k, modeltype in enumerate(models):
        last_epochs = np.zeros([n_folds, 2])
        accs = np.zeros([n_folds, 3])  # accuracy table for train/val/test

        # For the number of folds, pick the events to use
        for i in tqdm(range(n_folds)):
            test_sbj = sbj_inds_all_test[i]
            val_sbj = sbj_inds_all_val[i]
            train_sbj = sbj_inds_all_train[i]
            print('Train subjects:', train_sbj, 'Validation subjects:', val_sbj, 'Test subjects:', test_sbj)

            # Only need to determine train/val/test inds for first modeltype used
            if k == 0:
                # Find train/val/test indices (test inds differ depending on if test_day is specified or not)
                # Note that subject_data_inds will balance number of trials across classes
                train_inds, val_inds, test_inds = [], [], []
                if test_day is None:
                    test_inds = subject_data_inds(np.full(1, test_sbj), sbj_order, labels_unique, i,
                                                  'test_inds', half_n_evs_test, y, sp, n_folds, test_inds,
                                                  overwrite)
                else:
                    test_inds = subject_data_inds(np.full(1, test_sbj), sbj_order_test_load, labels_unique, i,
                                                  'test_inds', half_n_evs_test, y_test_orig, sp, n_folds, test_inds,
                                                  overwrite)
                val_inds = subject_data_inds(val_sbj, sbj_order, labels_unique, i,
                                             'val_inds', half_n_evs, y, sp, n_folds, val_inds, overwrite)
                train_inds = subject_data_inds(train_sbj, sbj_order, labels_unique, i,
                                               'train_inds', half_n_evs, y, sp, n_folds, train_inds, overwrite)
                train_inds_folds.append(train_inds)
                val_inds_folds.append(val_inds)
                test_inds_folds.append(test_inds)
            else:
                train_inds = train_inds_folds[i]
                val_inds = val_inds_folds[i]
                test_inds = test_inds_folds[i]

            # Now that we have the train/val/test event indices, generate the data for the models
            X_train = X[train_inds, ...]
            Y_train = y[train_inds]
            sbj_order_train = sbj_order[train_inds]
            X_validate = X[val_inds, ...]
            Y_validate = y[val_inds]
            sbj_order_validate = sbj_order[val_inds]
            if test_day is None:
                X_test = X[test_inds, ...]
                Y_test = y[test_inds]
                sbj_order_test = sbj_order[test_inds]
            else:
                X_test = X_test_orig[test_inds, ...]
                Y_test = y_test_orig[test_inds]
                sbj_order_test = sbj_order_test_load[test_inds]

            # Reformat data size for NN fitting
            # Y_train = to_categorical(Y_train - 1)
            Y_train -= 1
            X_train = np.expand_dims(X_train, 1)
            # Y_validate = to_categorical(Y_validate - 1)
            Y_validate -= 1
            X_validate = np.expand_dims(X_validate, 1)
            # Y_test = to_categorical(Y_test - 1)
            Y_test -= 1
            X_test = np.expand_dims(X_test, 1)
            proj_mat_out2 = np.expand_dims(proj_mat_out, 1)

            # Fit NN model using Keras
            chckpt_path = sp + 'checkpoint_gen_' + modeltype + '_fold' + str(i) + save_suffix + '.h5'
            accs_lst, last_epoch_tmp = cnn_model(X_train, Y_train, X_validate, Y_validate, X_test, Y_test,
                                                 chckpt_path, modeltype, proj_mat_out2, sbj_order_train,
                                                 sbj_order_validate, sbj_order_test, nROIs=nROIs,
                                                 nb_classes=nb_classes, dropoutRate=dropoutRate,
                                                 kernLength=kernLength,
                                                 F1=F1, D=D, F2=F2, dropoutType=dropoutType,
                                                 kernLength_sep=kernLength_sep,
                                                 patience=patience,
                                                 do_log=do_log,
                                                 epochs=epochs, compute_val=compute_val, ecog_srate=ecog_srate)

            # Store train/val/test accuracies, and last epoch
            for ss in range(3):
                accs[i, ss] = accs_lst[ss]

            last_epochs[i, :] = last_epoch_tmp

        # Save accuracies for all folds for one type of model
        np.save(sp + 'acc_gen_' + modeltype + '_' + str(n_folds) + save_suffix + '.npy', accs)
        np.save(sp + 'last_training_epoch_gen_tf' + modeltype + '_' + str(n_folds) + save_suffix + '.npy',
                last_epochs)

        result.append(accs[:, 1].mean())

    # Returns average validation accuracy for hyperparameter tuning (will be for last model_type only)
    return result
