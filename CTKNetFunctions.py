#!/usr/bin/env python
# coding: utf-8

# # CerealTimeKillersNet: Deep neural network for emotional states predictions from EEG data.

# ## Package setup

# In[1]:


# Basic packages
import copy
import time
import random
import numpy as np
import pandas as pd

# Math packages
import math
from scipy.signal import spectrogram

# Plot packages
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# PyTorch packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler, random_split

# ML Packages
from sklearn.model_selection import KFold


# ## Basic functions setup

# In[2]:


# Set random seed
# Executing `set_seed(seed=seed)` you are setting the seed to ensure reproducibility.
# for DL its critical to set the random seed so that students can have a
# baseline to compare their results to expected results.
# Read more here: https://pytorch.org/docs/stable/notes/randomness.html

def set_seed(seed = None, seed_torch = True):
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')

# In case that `DataLoader` is used
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# In[3]:


# @title Set device (GPU or CPU). Execute `set_device()` especially if torch modules used.
# inform the user if the notebook uses GPU or CPU.

def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: For this notebook to perform best, "
            "if possible, in the menu under `Runtime` -> "
            "`Change runtime type.`  select `GPU` ")
    else:
        print("GPU is enabled in this notebook.")

    return device


# In[4]:


# plot settings

def plot_loss_accuracy(train_loss, val_loss, test_loss, train_acc, val_acc, test_acc,
                       chance = 0.25):
    epochs = len(train_loss)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(list(range(epochs)), train_loss, label = 'Training')
    ax1.plot(list(range(epochs)), val_loss, label = 'Validation')
    ax1.plot(list(range(epochs)), test_loss, label = 'Testing')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Epoch vs Loss')
    ax1.legend()

    ax2.plot(list(range(epochs)), train_acc, label = 'Training')
    ax2.plot(list(range(epochs)), val_acc, label = 'Validation')
    ax2.plot(list(range(epochs)), test_acc, label = 'Testing')
    ax2.plot(list(range(epochs)), [chance * 100] * epochs, 'k--', label = 'Baseline')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 100])
    ax2.set_title('Epoch vs Accuracy')
    ax2.legend()
    
    fig.set_size_inches(15.5, 5.5)
    plt.show()
    


# In[5]:


# Norm
def calculate_frobenius_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.sum(param ** 2)
    norm = norm ** 0.5
    return norm

def L1_norm(model):
    return sum(p.abs().sum() for p in model.parameters())

def L2_norm(model):
    return sum((p**2).sum() for p in model.parameters())


# In[6]:


# Maximum label comparison for accracy

def maximum_extraction(tens, eplison = 1e-8):
    l_index = []
    for i in range(tens.shape[0]):
        label = tens[i].detach().numpy()
        l = []
        for j in range(len(label)):
            if label[j] > max(label) - eplison:
                l.append(j)
        l_index.append(l)
    return l_index
    
def maximum_comparison(list1, list2): # list2 is supposed to be real labels with multiple maximum values
    tot = 0
    for i in range(len(list1)):
        for j in range(len(list2[i])):
            list2[i][j] = round(list2[i][j])
        for j in range(len(list1[i])):
            if int(list1[i][j]) in list2[i]:
                tot += 1
                break
    return tot


# ## DataLoader setup

# In[7]:


class CerealTimeKillersDataset(Dataset):
    """Spectrogram dataset for torch"""

    def __init__(self, df, transform = None):
        self.ori_dataframe = df
        self.transform = transform

    def __len__(self):
        return len(self.ori_dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        spectrogram = self.ori_dataframe.iloc[idx, -1]
        spectrogram = torch.tensor(spectrogram)
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        labels = self.ori_dataframe.iloc[idx, :-2]
        labels = torch.tensor(labels).type(torch.FloatTensor)
        if self.transform:
            labels = self.transform(labels)
        
        quadrants = self.ori_dataframe.iloc[idx, -2]
        quadrants = torch.tensor(quadrants).type(torch.FloatTensor)
        if self.transform:
            quadrants = self.transform(quadrants)
        
        return (spectrogram, labels, quadrants)
    


# In[8]:


def get_specgram(df, labels, winlen = None, stride = 1, nperseg = 256, fs = 129):
    """
    Spectrogram from EEG data
    
    Inputs:
    df (pandas.DataFrame): EEG dataframe
    labels (CerealTimeKillersLabels): Electrode labels used for model prediction
    winlen (None/int): Time window for input sampling (for the whole timepoints, Default is None)
    stride (int): Temporal leap for input sampling (Default is 1)
    nperseg (int): N per seg of spectrogram (Default is 256)
    fs (int): Framerate of spectrogram (Default is 128)
    
    Returns:
    data (np.array): EEG data spectrogram with [samplepoint, frequency, time, channel]
    """
    
    # Load selected electrodes
    df = pd.DataFrame(df, columns = labels)
    d = np.array(df, dtype = float) # Switching from pandas to numpy array as this might be more comfortable for people
    
    full_spec = []
    for idx, d2 in enumerate(d.T):
        _, _, Sxx = spectrogram(d2, nperseg = nperseg, fs = fs)
        full_spec.append(Sxx)
        
    #DIMENSIONS OF FULL_SPEC WITHOUT WINDOWING (I.E. FULL WINDOWING)
    #DIMENSION 1: 1                      - FOR DIMENSIONAL CONSISTENCY
    #DIMENSION 2: CHANNELS  (DEFAULT=14) - MIGHT CHANGE (SO NOT REALLY DEFAULT BUT OK)
    #DIMENSION 3: FREQUENCY (DEFAULT=129)
    #DIMENSION 4: TIME      (DEFAULT=170) - MIGHT CHANGE AS WELL OK - WE ARE WORKING ON IT
    
    full_spec = np.vstack([full_spec])
    full_spec = np.moveaxis(full_spec, 0, 0)
    if winlen == None:
        return np.array([full_spec])
    
    i = 0
    full_spec_wind = []
    # STRICK THE FOLLOWING LOOP ON THE TIME (WINDOW) DIMENSION!
    while i * stride + winlen < full_spec.shape[2]:
        full_spec_wind.append(full_spec[: , : , i * stride : i * stride + winlen])
        i += 1
    
    #DIMENSIONS OF FULL_SPEC WITH WINDOWING    (FULL_SPEC_WIND) 
    #DIMENSION 1: SAMPLE    (NO DEFAULT - SORRY)
    #DIMENSION 2: CHANNELS  (DEFAULT=14) - MIGHT CHANGE (SO NOT REALLY DEFAULT BUT OK)
    #DIMENSION 3: FREQUENCY (DEFAULT=129)
    #DIMENSION 4: WINDOWS   (DEFAULT=1)
    
    full_spec_wind = np.array(full_spec_wind)
    return full_spec_wind


# In[9]:


def CerealTimeKillersDataLoader(dir_base, label_class, label_range, 
                                dataset_mix = True, 
                                winlen = None, stride = 1, nperseg = 256, fs = 129,
                                transform = None):
    """
    Cereal Time Killers Data Loader
    
    Inputs:
    dir_base (str): Working space dictionary
    label_class (CerealTimeKillersLabels): Labels used for model prediction
    label_range (1*2 list): The [min, max] of emotional states for transformation
    dataset_mix (bool): Whether to allow between-subject and between-game dataset mixture (Default is True)
    winlen (None/int): Time window for input sampling (for the whole timepoints, Default is None)
    stride (int): Temporal leap for input sampling (Default is 1)
    nperseg (int): N per seg of spectrogram (Default is 256)
    fs (int): Framerate of spectrogram (Default is 128)
    transform (torchvision.transforms.transforms.Compose): Torch transormfation (Default is None)
    
    Returns:
    FullDataset (CerealTimeKillersDatase list): full data with EEG spectrogram and fixed labels (information and/or emotional states) in CerealTimeKillersLabels
        FullDataset[i]: ith datapoint of [spectrogram, labels, quadrants]
    DataSize (Tuple): Data size for single point as (Input size as tuple, Output size as int)
    ExpIndex (pandas.DataFrame): Corresponsing ['subject', 'game'] index with shared row indices from FullDataset
    """
    
    specgram_name = 'full_specgram_1'
    
    # Load label & EEG data
    labels_df = pd.read_csv(f'{dir_base}GameLabels.csv')
    spec_df = pd.DataFrame(columns = label_class.fixed + ['emotion', specgram_name], dtype = float)
    index_df = pd.DataFrame(columns = ['subject', 'game'], dtype = int)
    
    # Create spectrogram dataframe
    for idx in range(labels_df.shape[0]): 
        
        # Load info and fixed labels
        subject = int(labels_df['subject'].iloc[idx])
        game = int(labels_df['game'].iloc[idx])
        fixed_labels = labels_df[label_class.fixed].iloc[idx]
        fixed_labels = list(np.array(np.array(fixed_labels, dtype = 'float') - label_range[0]) / (label_range[1] - label_range[0]))
        
        # Maximum quadrant emotion labels
        quadrant_labels = labels_df[label_class.quadrant].iloc[idx]
        quadrant_labels = list(np.array(quadrant_labels, dtype = 'float'))
        
        # You can also just paste in the Directory of the csv file - on windows you may have to change the slash direction
        DirComb = f'{dir_base}GAMEEMO/(S{str(subject).zfill(2)})/Preprocessed EEG Data/.csv format/S{str(subject).zfill(2)}G{str(game)}AllChannels.csv'
        CsvSpec = pd.read_csv(DirComb, sep = ',')
        
        # Get EEG spectrogram
        spec_EEG = get_specgram(CsvSpec, label_class.electrode, 
                                winlen = winlen, stride = stride, nperseg = nperseg, fs = fs)
        
        # Add new data to dataframe
        new_spec_list, new_index_list = list(), list()
        if dataset_mix:
            for i in range(spec_EEG.shape[0]):
                new_spec_list.append(fixed_labels + [quadrant_labels] + [spec_EEG[i]])
                new_index_list.append([subject, game])
        else:
            new_spec_list.append(fixed_labels + [quadrant_labels] + [spec_EEG])
            new_index_list.append([subject, game])
        
        # Update dataframe
        new_spec_df = pd.DataFrame(new_spec_list, columns = label_class.fixed + ['emotion', specgram_name], dtype = float)
        spec_df = pd.concat([spec_df, new_spec_df], ignore_index = True)    
        new_index_df = pd.DataFrame(new_index_list, columns = ['subject', 'game'], dtype = int)
        
        index_df = pd.concat([index_df, new_index_df], ignore_index = True)
    
    # Output
    final_df = CerealTimeKillersDataset(df = spec_df, transform = transform)
    data_size = (tuple(final_df[0][0].shape), tuple(final_df[0][1].shape))

    return final_df, data_size, index_df


# In[10]:


def CerealTimeKillersDataSplitter(full_dataset, exp_index, 
                                  allocation_test = None, 
                                  test_ratio = 0.2, target_test = [], k_folds = 10, 
                                  batch_size_train = 16, batch_size_test = 32, 
                                  seed = 0, generator = None):
    """
    Cereal Time Killers Data Splitter
    
    Inputs:
    full_dataset (CerealTimeKillersDataset): full data with EEG spectrogram and experimental labels (information and emotional states)
        full_dataset[i]: ith data for a specific subject and game of {'spectrogram': spectrogram, 'labels': labels}
    exp_index (pandas.DataFrame): Corresponsing ['subject', 'game', 'emotion'] with shared row indices from full_dataset
    allocation_test (None/str): Which to be based for allocating testing dataset (Default is None) # [None, 'subject', 'game']
    test_ratio (float) Proportion of data used for testing when Allocation_test == None (Default is 0.2)
    target_test (list): Int list for allocating corresponding game/subject as testing dataset when Allocation_test != None (Default is [])
    k_folds (int): Number for K-folds for training vs validation (Default is 10)
    batch_size_train (int): Number of examples per minibatch during training (Default is 16)
    batch_size_test (int): Number of examples per minibatch during validation/testing (Default is 1)
    seed (int): Random seed for reproducibility (Default is 0)
    generator (torch._C.Generator): Torch generator for reproducibility (Default is None)
    
    Returns:
    SplittedDataset (dict): Full dataset splitted in {'train': training, 'val': validation, 'test': testing}
        SplittedDataset['train'][fold].dataset (CerealTimeKillersDataset): Training dataset in nth fold
        SplittedDataset['val'][fold].dataset (CerealTimeKillersDataset): Validation dataset in nth fold
        SplittedDataset['test'].dataset (CerealTimeKillersDataset): Testing dataset outside folds
    SplittedDataLength (dict): Length of dataset in {'train': training, 'val': validation, 'test': testing}
    """
    
    # Split into train/val and test datasets
    train_set_index, test_set_index = list(), list()
    if allocation_test == None:
        test_size = int(test_ratio * len(full_dataset))
        train_size = len(full_dataset) - test_size
        train_set_orig, test_set_orig = random_split(full_dataset, 
                                                     [train_size, test_size], 
                                                     generator = generator)
    elif (allocation_test == 'subject') or (allocation_test == 'game'):
        train_set_index = exp_index[~exp_index[allocation_test].isin(target_test)].index.tolist()
        test_set_index = exp_index[exp_index[allocation_test].isin(target_test)].index.tolist()
        train_set_orig = Subset(full_dataset, train_set_index)
        test_set_orig = Subset(full_dataset, test_set_index)
    else:
        print("Allocate testing dataset based on one of the 'Subject', 'Game', or None.")
        return None
    
    # Test dataset loader
    test_loader = DataLoader(test_set_orig,
                             batch_size = batch_size_test,
                             num_workers = 0,
                             generator = generator)
    
    # K-fold Cross Validator
    train_loader, val_loader = [[]] * k_folds, [[]] * k_folds
    kfold = KFold(n_splits = k_folds, shuffle = True, random_state = seed)
    for fold, (train_i, val_i) in enumerate(kfold.split(train_set_orig)):
        
        # Sample train/test dataset from indices
        train_sampler = SubsetRandomSampler(train_i, generator = generator)
        val_sampler = SubsetRandomSampler(val_i, generator = generator)
        
        # Train/Validation dataset loader
        train_loader[fold] = DataLoader(train_set_orig,
                                        sampler = train_sampler,
                                        batch_size = batch_size_train,
                                        num_workers = 0,
                                        generator = generator)
        val_loader[fold] = DataLoader(train_set_orig,
                                      sampler = val_sampler,
                                      batch_size = batch_size_test,
                                      num_workers = 0,
                                      generator = generator)
    
    # return datasplitter
    data_loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    data_length = {'train': len(train_sampler), 'val': len(val_sampler), 'test': len(test_set_orig)}
    return data_loader, data_length


# ## Neural network setup

# In[11]:


def train(args, model, train_loader, optimizer = None, criterion = nn.MSELoss()):
    
    model.train()

    for (data, target, quadrant) in train_loader:
        data, target = data.type(torch.float).to(args['device']), target.type(torch.float).to(args['device'])
        optimizer.zero_grad() 
        output = model(data)
        
        loss = criterion(output, target) + args['l1'] * L1_norm(model) + args['l2'] * L2_norm(model)
        loss.backward()
        
        optimizer.step()


# In[12]:


def test(args, label, model, test_loader, is_2D = False, criterion = nn.MSELoss()):
    
    model.eval()
    
    eval_loss = 0.0
    acc = 0.0
    total = 0
    with torch.no_grad():
        for (data, target, quadrant) in test_loader:
            data = data.type(torch.float).to(args['device'])
            target = target.type(torch.float).to(args['device'])
            quadrant = quadrant.type(torch.float).to(args['device'])
            output = model(data)
            
            loss = criterion(output, target)
            eval_loss += loss.item()
            
            if not is_2D:
                predicted = maximum_extraction(output)
                labels = maximum_extraction(target)
            else:
                predicted = emotion_transformation(output, label)
                # labels = maximum_extraction(quadrant)
                labels = emotion_transformation(target, label)
            acc += maximum_comparison(predicted, labels)
            total += target.size(0)
            
    return eval_loss / len(test_loader), acc * 100 / total


# In[13]:


def simulation(args, label, model, train_loader, val_loader, test_loader, is_2D = False,
               optimizer = None, criterion = nn.MSELoss()):
    
    model = model.to(args['device'])
    
    val_loss_list, train_loss_list, test_loss_list = [], [], []
    val_acc_list, train_acc_list, test_acc_list = [], [], []
    param_norm_list = []
    best_loss = 100
    for epoch in tqdm(range(args['epochs'])):
        
        train(args, model, train_loader, optimizer = optimizer, criterion = criterion)
        param_norm = calculate_frobenius_norm(model)
        
        train_loss, train_acc = test(args, label, model, train_loader, is_2D = is_2D, criterion = criterion)
        val_loss, val_acc = test(args, label, model, val_loader, is_2D = is_2D, criterion = criterion)
        test_loss, test_acc = test(args, label, model, test_loader, is_2D = is_2D, criterion = criterion)
        
        if (val_loss < best_loss) or (epoch == 0):
            best_loss = val_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            wait = 0
        else:
            wait += 1
        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        param_norm_list.append(param_norm)
        
        if wait > args['patience']:
            print('Early stopped on epoch', best_epoch)
            break
        
        if ((epoch + 1) % 10 == 0) or (epoch + 1 == args['epochs']):
            print('-----Epoch ', epoch + 1, '/', args['epochs'])
            print('Train/Val/TEST MSE:', train_loss, val_loss, test_loss)
            print('Train/Val/TEST Accuracy:', train_acc, val_acc, test_acc)
    
    plot_loss_accuracy(train_loss_list, val_loss_list, test_loss_list, 
                       train_acc_list, val_acc_list, test_acc_list)

    return (train_loss_list, val_loss_list, test_loss_list), (train_acc_list, val_acc_list, test_acc_list), param_norm_list, best_model, best_epoch


# ## Emotion Transformation

# In[11]:


# 2D emotions ['valence', 'arousal'] to 4 quadrant emotions ['boring', 'horrible', 'calm', 'funny']
def emotion_transformation(pred, label, c = 0.5):
    # Quadrant I:   ('valence' >= c, 'arousal' >= c) --> 'funny' 
    # Quadrant II:  ('valence' >= c, 'arousal' <= c) --> 'calm'
    # Quadrant III: ('valence' <= c, 'arousal' <= c) --> 'boring'
    # Quadrant IV:  ('valence' <= c, 'arousal' >= c) --> 'horrible'
    ans = []
    pred = pred.detach().numpy()
    i, j = 0, 1 # index of 'valence' and 'arousal' in CerealTimeKillersLabels.prediction
    
    for k in range(pred.shape[0]): # batch
        anss = []
        if (pred[k, i] >= c) and (pred[k, j] >= c):
            anss.append(3) # index of 'funny' in CerealTimeKillersLabels.quadrant
        if (pred[k, i] >= c) and (pred[k, j] <= c):
            anss.append(2) # index of 'calm' in CerealTimeKillersLabels.quadrant
        if (pred[k, i] <= c) and (pred[k, j] >= c):
            anss.append(1) # index of 'horrible' in CerealTimeKillersLabels.quadrant
        if (pred[k, i] <= c) and (pred[k, j] <= c):
            anss.append(0) # index of 'boring' in CerealTimeKillersLabels.quadrant
        ans.append(anss)
        
    return ans

