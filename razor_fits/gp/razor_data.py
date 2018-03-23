import os
import itertools

import numpy as np
import pandas as pd
import torch.utils.data

### UTILITY FUNCTIONS

def get_boxes():
    return ['DiJet', 'MultiJet', 'SevenJet', 
            'LeptonMultiJet', 'LeptonSevenJet']

def is_lepton_box(box):
    return 'Lepton' in box

def get_btag_bins(box):
    if box == 'DiJet':
        return range(3)
    return range(4)

def get_mr_min(box):
    """Returns the lower edge of the MR range for the given box."""
    if is_lepton_box(box):
        return 550
    return 650

def get_rsq_min(box):
    """Returns the lower edge of the Rsq range for the given box."""
    if is_lepton_box(box):
        return 0.20
    return 0.30

def get_signal_norm(sms):
    """
    Gets xsec * intlumi for the given signal model
    """
    xsecs = {
            'T1':{1800: 0.00276133, 1000: 0.325388},
            'T2':{1000: 0.00615134}
            }
    nevents = {
            'T1':{1800: 15000, 1000: 150000},
            'T2':{1000: 20000}
            }
    # our TTJets MC sample corresponds to about 500 ifb of data
    lumi = 500000
    model, m, _ = sms.split('_')
    model = model[:2]
    m = int(m)
    return xsecs[model][m] * lumi / nevents[model][m]

### PANDAS FUNCTIONS

def get_data_dir():
    return os.path.realpath(os.path.dirname(__file__))+'/data/'

def load_mrrsq_data(box, nb, proc='TTJets1L'):
    """
    Locates the data file for the appropriate box
    and returns a Pandas DataFrame containing the
    MR and Rsq values in each event with #b-tags = nb.
    """
    data_dir = get_data_dir()
    file_name = data_dir+'/{}_{}_Razor2016_MoriondRereco.csv'.format(proc, box)
    df = pd.read_csv(file_name)
    df_reduced = df[df['nBTaggedJets'] == nb][['MR', 'Rsq']]
    rsq_cut = get_rsq_min(box)
    return df_reduced[df_reduced['Rsq'] > rsq_cut]

def bin_edges_to_centers(edges):
    return (edges[:-1] + edges[1:]) / 2.0

def df_to_hist_1d(df, num_mr_bins, mr_min, mr_max):
    vals = df['MR'].values
    counts, edges = np.histogram(vals, num_mr_bins, range=(mr_min, mr_max))
    bin_centers = bin_edges_to_centers(edges)
    return bin_centers, counts

def df_to_hist_2d(df, num_mr_bins, mr_min, mr_max,
        num_rsq_bins, rsq_min, rsq_max):
    mr_vals = df['MR'].values
    rsq_vals = df['Rsq'].values
    counts, mr_edges, rsq_edges = np.histogram2d(
            mr_vals, rsq_vals, (num_mr_bins, num_rsq_bins),
            range=((mr_min, mr_max), (rsq_min, rsq_max)))
    mr_centers = bin_edges_to_centers(mr_edges)
    rsq_centers = bin_edges_to_centers(rsq_edges)
    bin_centers = np.asarray([x for x in itertools.product(mr_centers, rsq_centers)])
    return bin_centers, counts.flatten()

def binned_data_to_df(binned):
    bins = binned['u'].numpy()
    mr_centers = bins[:, 0]
    rsq_centers = bins[:, 1]
    counts = binned['y'].numpy()
    return pd.DataFrame({'mr_center': mr_centers, 'rsq_center': rsq_centers,
        'counts': counts})


### DATASET CLASSES

class RazorDataset(torch.utils.data.Dataset):
    """
    This class loads the razor inclusive data for a specific box
    and returns it in binned format.
    """

    def __init__(self, box, nb, num_mr_bins=100, mr_max=4000,
            proc='TTJets1L'):
        self.box = box
        self.nb = nb
        self.proc = proc

        self.num_mr_bins = num_mr_bins
        self.mr_min = get_mr_min(box)
        self.mr_max = mr_max

        unbinned = load_mrrsq_data(box, nb, proc=proc)
        self.bin_centers, self.counts = self.convert_to_binned(unbinned)

        # normalize signal samples
        if 'T1' in proc or 'T2' in proc:
            self.counts = self.counts * get_signal_norm(proc)

    def convert_to_binned(self, unbinned):
        raise NotImplementedError(
                "Please use subclasses Razor1DDataset or Razor2DDataset")

    def __len__(self):
        return len(self.bin_centers)

    def __getitem__(self, idx):
        sample = {'u': self.bin_centers[idx], 'y': self.counts[idx]}
        return sample

        
class Razor1DDataset(RazorDataset):
        
    def convert_to_binned(self, unbinned):
        return df_to_hist_1d(unbinned, self.num_mr_bins, 
                self.mr_min, self.mr_max)


class Razor2DDataset(RazorDataset):

    def __init__(self, box, nb, num_mr_bins=20, mr_max=4000,
            num_rsq_bins=20, rsq_max=1.5):
        self.num_rsq_bins = num_rsq_bins
        self.rsq_min = get_rsq_min(box)
        self.rsq_max = rsq_max
        super(Razor2DDataset, self).__init__(box, nb, num_mr_bins, mr_max)

    def convert_to_binned(self, unbinned):
        return df_to_hist_2d(unbinned, self.num_mr_bins, 
                self.mr_min, self.mr_max, self.num_rsq_bins, 
                self.rsq_min, self.rsq_max)


### DATA LOADERS

def get_loader(dataset, batch_size=None, shuffle=False):
    if batch_size is None:
        batch_size = len(dataset)
    return torch.utils.data.DataLoader(dataset, 
            batch_size=batch_size, shuffle=shuffle)

def get_binned_data_1d(num_mr_bins, mr_max=2000, proc='TTJets1L'):
    """
    Returns a dictionary containing MR-Rsq datasets for all
    boxes and b-tag categories.
    """
    boxes = get_boxes()
    nbtags = {box: get_btag_bins(box) for box in boxes}

    datasets = {}
    loaders = {}
    binned_data = {}
    for box in boxes:
        datasets[box] = {nb: Razor1DDataset(
            box, nb, num_mr_bins, mr_max, proc=proc) for nb in nbtags[box]}
        loaders[box] = {nb: get_loader(dataset) 
                for nb, dataset in datasets[box].items()}
        binned_data[box] = {nb: iter(loader).next()
                for nb, loader in loaders[box].items()}
        # convert DoubleTensors to FloatTensors to avoid incompatibility
        for nb in binned_data[box]:
            binned_data[box][nb]['u'] = binned_data[box][nb]['u'].float()
            binned_data[box][nb]['y'] = binned_data[box][nb]['y'].float()
    return binned_data

def get_binned_data_2d(num_mr_bins, num_rsq_bins, mr_max=2000, rsq_max=1.0):
    boxes = get_boxes()
    nbtags = {box: get_btag_bins(box) for box in boxes}

    datasets = {}
    loaders = {}
    binned_data = {}
    for box in boxes:
        datasets[box] = {nb: Razor2DDataset(
            box, nb, num_mr_bins, mr_max, 
            num_rsq_bins, rsq_max) for nb in nbtags[box]}
        loaders[box] = {nb: get_loader(dataset) 
                for nb, dataset in datasets[box].items()}
        binned_data[box] = {nb: iter(loader).next()
                for nb, loader in loaders[box].items()}
    return binned_data
