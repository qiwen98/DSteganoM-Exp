import torch
from torchmetrics.functional import peak_signal_noise_ratio
torch.manual_seed(0)

def customized_loss(S_prime, C_prime, S, C, B=1):
    ''' Calculates loss specified on the paper.'''

    loss_cover = torch.nn.functional.mse_loss(C_prime, C)
    # loss_cover = torch.mean((C_prime - C) ** 2)
    loss_secret = torch.nn.functional.mse_loss(S_prime, S)
    # loss_secret = torch.mean((S_prime - S) ** 2)
    loss_all = (loss_cover + B * loss_secret)
    return loss_all, loss_cover, loss_secret

def customized_PSNR(S_prime, C_prime, S, C, B=1):
    ''' Calculates PSNR based on the PSNR method -2 .'''
    PSNR_cover = peak_signal_noise_ratio(C_prime, C)
    # loss_cover = torch.mean((C_prime - C) ** 2)
    PSNR_secret = peak_signal_noise_ratio(S_prime, S)
    # loss_secret = torch.mean((S_prime - S) ** 2)
    PSNR_all = (PSNR_cover + B * PSNR_secret)/2
    return PSNR_all, PSNR_cover, PSNR_secret

def calculate_per_part_loss(S_prime, C_prime, S, C, dataset):
    '''calculate mse loss per part, eg, thumb, index, middle, ring, pinky'''
    loss = torch.nn.MSELoss(reduce=False)

    loss_cover= loss(C_prime, C).squeeze() # N,C,P,T -> CPT
    loss_secret = loss(S_prime, S).squeeze() #/ N,C,P,T ->CPT

    #print(loss_cover.shape)

    # loss_cover= loss_cover.gpu().detach().numpy()
    # loss_secret = loss_secret.gpu().detach().numpy()

    loss_cover = loss_cover.transpose(1,2).mean(axis=-1) # C,T,P -> C,P
    loss_secret = loss_secret.transpose(1,2).mean(axis=-1) #C,T,P -> C,P

    loss_cover = loss_cover.mean(axis=0) # C,P->P
    loss_secret = loss_secret.mean(axis=0)  # C,P->P
    #print(loss_cover.shape)

    loss_cover= loss_cover.cpu().detach().numpy()
    loss_secret = loss_secret.cpu().detach().numpy()


    # per_part_dict


    per_part_dict=np.zeros(5)
    if dataset == "CMU":
        # hips,left legs
        per_part_dict[0]= (np.mean(loss_cover[:7])+np.mean(loss_secret[:7]))/2 #0,7
        # right legs
        per_part_dict[1] = (np.mean(loss_cover[7:13])+np.mean(loss_secret[7:13]))/2  #7:13
        # lower back spine
        per_part_dict[2] = (np.mean(loss_cover[13:20])+np.mean(loss_secret[13:20]))/2  #13:20
        # left shoulder
        per_part_dict[3] = (np.mean(loss_cover[20:27])+np.mean(loss_secret[20:27]))/2 #20:27
        # right shoulder
        per_part_dict[4] = (np.mean(loss_cover[27:-1])+np.mean(loss_secret[27:-1]))/2 #27:-1


    else:
        # thumb
        per_part_dict[0]= (np.mean(loss_cover[:5])+np.mean(loss_secret[:5]))/2
        # index
        per_part_dict[1] = (np.mean(loss_cover[5:9])+np.mean(loss_secret[5:9]))/2  #5:9
        # middle
        per_part_dict[2] = (np.mean(loss_cover[9:13])+np.mean(loss_secret[9:13]))/2  #9:13
        #ring
        per_part_dict[3] = (np.mean(loss_cover[13:17])+np.mean(loss_secret[13:17]))/2 #13:17
        #pinky
        per_part_dict[4] = (np.mean(loss_cover[17:-1])+np.mean(loss_secret[17:-1]))/2 #17:-1
    return per_part_dict



def corrupt(X, sigma_o, sigma_s, beta):
    if X.dim() != 3:
        raise Exception("Sorry, the X dimension should be 3")

    T, N, _ = X.size()

    # normal dis # shape T
    alpha_o = torch.empty(T, N).cuda()
    torch.nn.init.normal_(alpha_o, 0, sigma_o)

    alpha_s = torch.empty(T, N).cuda()
    torch.nn.init.normal_(alpha_s, 0, sigma_s)

    # print(alpha_s)
    # Mask #shape T*N
    Xo = torch.bernoulli(torch.clamp(torch.abs(alpha_o),
                                     max=2 * sigma_o))  # take the min of |x| and sigma #calmp max means ,if the value greater than max then just take the max, === take min
    Xs = torch.bernoulli(torch.clamp(torch.abs(alpha_s), max=2 * sigma_s))

    # print(Xs)
    # shift # shape T*N*3
    Xv = torch.empty(T, N, 3).cuda()
    torch.nn.init.uniform_(Xv, -beta, beta)

    return (X + Xs[:, :, None] * Xv) * (1 - Xo[:, :, None])

def filter_OutZero(data):
    data = [i for i in data if i.sum() != 0]
    return data

class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from functools import lru_cache
from torch.utils.data.sampler import SubsetRandomSampler
from loguru import logger


class H5_Dataset(Dataset):

    def __init__(self, hdf5_file, transform=None):
        self.dataset = h5py.File(hdf5_file, 'r')
        self.data = self.dataset["300_frames_with_30fps/feature"]
        # self.min = np.min(self.data)
        # self.max = np.max(self.data)
        self.max=  6.3129#5.5997#20.035435
        self.min = -6.6153#-5.6810#-20.501478
        self.newmin = -1
        self.newmax = 1
        self.transform = transform
        # self.label = xxx

    def __getitem__(self, index):
        datum = self.data[index]

        datum = datum.transpose(1, 2, 0)
        # print(datum.shape)
        datum = torch.from_numpy(datum).float()

        # in_transform = transforms.Compose([transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])
        if self.transform is not None:
            datum = self.transform(datum)

        datum = (datum - self.min) / (self.max - self.min) * (self.newmax - self.newmin) + self.newmin  ## rescale
        return datum

    def __len__(self):
        return len(self.data)


class DataSplit:

    def __init__(self, dataset, test_train_split=0.85, val_train_split=0.15, shuffle=False):
        self.dataset = dataset

        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        train_indices, self.test_indices = self.indices[:test_split], self.indices[test_split:]
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))

        self.train_indices, self.val_indices = train_indices[: validation_split], train_indices[validation_split:]

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):
        logger.info('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        logger.info('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                        sampler=self.train_sampler, shuffle=False,
                                                        num_workers=num_workers, drop_last=True)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        logger.info('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.val_sampler,
                                                      shuffle=False, num_workers=num_workers, drop_last=True)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        logger.info('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                       sampler=self.test_sampler, shuffle=False,
                                                       num_workers=num_workers, drop_last=True)
        return self.test_loader


class Custom_Dataset(Dataset):

    def __init__(self, hdf5_file, MTMdataset_features, transform=None):
        self.CMUdataset = h5py.File(hdf5_file, 'r')
        self.CMUdata = self.CMUdataset["300_frames_with_30fps/feature"]  # CMU for cover
        self.MTMdata = MTMdataset_features  # MTM for secret
        # self.min = np.min(self.data)
        # self.max = np.max(self.data)
        self.max=  6.3129#5.5997#20.035435
        self.min = -6.6153#-5.6810#-20.501478
        self.newmin = -1
        self.newmax = 1
        self.transform = transform
        # self.label = xxx

    def __getitem__(self, index):
        CMUdatum = self.CMUdata[index]  # CMU for cover
        MTMdatum = self.MTMdata[index]  # MTM for secret

        CMUdatum = CMUdatum.transpose(1, 2, 0)
        # print(datum.shape)
        CMUdatum = torch.from_numpy(CMUdatum).float()

        # in_transform = transforms.Compose([transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])
        if self.transform is not None:
            CMUdatum = self.transform(CMUdatum)
            CMUdatum = (CMUdatum - self.min) / (self.max - self.min) * (
                        self.newmax - self.newmin) + self.newmin  ## rescale

        return CMUdatum, MTMdatum  # cover, secret

    def __len__(self):
        return min(len(self.CMUdata), len(self.MTMdata))