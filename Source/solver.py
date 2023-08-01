from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from os.path import exists, join
from os import makedirs
from dataloader import mtmLoader

from CNNModel import Encoder, CarrierDecoder, MsgDecoder
from tqdm.auto import tqdm, trange


import os
from os.path import join, basename
from collections import defaultdict
from experiment import Experiment

class Solver(object):
    def __init__(self, config):
        self.config = config
        # optimization hyperparams
        self.lr = config.lr
        self.lr_decay = config.lr_decay
        self.gamma_msg_loss = config.gamma_msg_loss
        self.opt_type = {'adam': torch.optim.Adam,
                         'sgd':  torch.optim.SGD,
                         'rms':  torch.optim.RMSprop}[config.opt]

        # training config
        self.num_iters = config.num_iters
        self.cur_iter = 0
        self.loss_type = config.loss_type
        # self.train_path = config.train_path
        # self.val_path = config.val_path
        # self.test_path = config.test_path
        self.batch_size = config.batch_size
        self.train_test_ratio = config.train_test_ratio
        self.val_test_ratio = config.val_test_ratio
        self.n_messages = config.n_messages

        self.dataset= config.dataset

        #corrupt info
        self.sigma_o = config.sigma_o
        self.sigma_s = config.sigma_s
        self.beta = config.beta

        # ## above complete
        # self.trim_start = {'yoho': int(2.0*8000),
        #                    'timit': int(0.6*16000)}[self.dataset]
        # if config.mode == 'sample':
        #     self.trim_start = 0
        # self.num_samples = int({'yoho': AUDIO_LEN * 8000,
        #                         'timit': AUDIO_LEN * 16000}[self.dataset])



        #model config for building (need to Pay attention here !!!!!)
        self.block_type = config.block_type
        self.enc_n_layers = config.enc_n_layers
        self.dec_c_n_layers = config.dec_c_n_layers
        self.dec_c_conv_dim = 3 + 3 + 64
        self.dec_m_conv_dim = 3
        self.dec_m_num_repeat = 8


        # create experiment
        self.experiment    = Experiment(config)

        self.num_workers        = config.num_workers
        self.device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.save_model_every   = config.save_model_every
        # self.sample_every       = config.sample_every
        self.print_every        = 10
        self.mode               = config.mode

        self.ckpt_dir = join(config.run_dir, 'ckpt')
        if self.mode == 'test':
            self.load_ckpt_dir = join(config.run_dir, 'ckpt')
        else:
            self.load_ckpt_dir = None

        # self.create_dirs()
        self.load_data()
        self.build_models()

        # self.stft.num_samples = self.num_samples
        torch.cuda.empty_cache()

        torch.autograd.set_detect_anomaly(True)

    def log_losses(self, losses, iteration=None):
        if iteration is None:
            iteration = self.cur_iter

        self.experiment.log_metric(losses, step=iteration)

    def create_dirs(self):
        makedirs(self.samples_dir, exist_ok=True)
        logger.info("created dirs")

    def load_data(self):
        train,test,val = {'MTM': mtmLoader(),
                  }[self.dataset]

        self.train_loader = DataLoader(train,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=self.num_workers,
                                       drop_last=True)
        self.test_loader = DataLoader(test,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      drop_last=True)
        self.val_loader = DataLoader(val,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=self.num_workers,
                                     drop_last=True)


        logger.info(f"loaded train ({len(train)}), val ({len(val)}), test ({len(test)})")

    def build_models(self):
        raise NotImplementedError

    def save_models(self, suffix=''):
        raise NotImplementedError

    def load_models(self, ckpt_dir):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def reset_grad(self):
        raise NotImplementedError

    def incur_loss(self, carrier, carrier_reconst, msg, msg_reconst):
        raise NotImplementedError

    def forward(self, carrier, msg):
        raise NotImplementedError

    def train_mode(self):
        logger.debug("train mode")
        self.mode = 'train'

    def eval_mode(self):
        logger.debug("eval mode")
        self.mode = 'test'

    def reconstruction_loss(self, input, target, type='mse'):
        if type == 'mse':
            loss = F.mse_loss(input, target)
        elif type == 'abs':
            loss = F.l1_loss(input, target)
        else:
            logger.error("unsupported loss function! reverting to MSE...")
            loss = F.mse_loss(input, target)
        return loss

    def train(self):
        # self.eval_mode()
        #  self.sample_examples(subdir=f"epoch_0")

        # start of training loop
        logger.info("start training...")
        epoch_it = trange(1, self.num_iters+1)
        lowest_cost = 100

        for epoch in epoch_it:
            lr = self.opt.param_groups[0]['lr']
            epoch_it.set_description(f"Epoch {epoch}, LR={lr}")
            epoch_loss = defaultdict(list)
            it = tqdm(self.train_loader)
            total_loss = 0
            self.cur_iter += 1
            self.train_mode()
            avg_carrier_loss, avg_msg_loss = 0, 0

            # inner epoch loop
            for idx, train_batch in enumerate(it):

                data = train_batch.float()
                # data = train_batch[0]


                # Saves secret images and secret covers
                train_covers = data[:len(data) // 2] # should be N,C,V,T
                train_secrets = [data[len(data) // 2:] ]# is a list

                # feedforward and suffer loss
                carrier_reconst, msg_reconst = self.forward(train_covers, train_secrets)
                loss, losses_log = self.incur_loss(train_covers, carrier_reconst, train_secrets, msg_reconst)

                self.reset_grad()
                loss.backward()
                self.step()


                # log epoch losses
                for k,v in losses_log.items():
                    epoch_loss[k].append(v)

                total_loss += loss


            # calc epoch stats
            for k,v in list(epoch_loss.items()):
                epoch_loss["train/ epoch_" + k] = np.mean(v)
                epoch_loss.pop(k)
            epoch_loss['lr'] = lr
            total_loss = total_loss/idx+1
            epoch_loss['train/ epoch total loss'] = total_loss

            #   epoch average loss
            logger.info(' Epoch [{0}], Average_loss: {1:.5f}, : loss-cover{2:.5f} :loss_secret{3:.5f}'.format(
                epoch, epoch_loss['train/ epoch total loss'], epoch_loss['train/ epoch_carrier_loss'], epoch_loss['train/ epoch_avg_msg_loss']))

            self.log_losses(epoch_loss, iteration=epoch)

            if total_loss < lowest_cost:
                lowest_cost = total_loss
                self.experiment.update_summary({'train/_lowest_cost': total_loss, 'train/_lowest_cost_epoch': epoch})
                self.save_models(suffix="epoch{0}_{1}_".format(epoch , self.block_type))

            if epoch % 10 == 0:
                # put everything in eval mode for sampling
                self.eval_mode()

                # save model every epoch
                self.save_models(suffix="epoch{0}_{1}_".format(epoch,self.block_type))

                # sample every epoch
                #  self.sample_examples(subdir=f"epoch_{epoch+1}")

                # run validation and log losses
                self.log_losses(self.test(mode='val'), iteration=epoch)

        logger.info("finished training!")

    # def snr(self, orig, recon):
    #     N = orig.shape[-1] * orig.shape[-2]
    #     orig, recon = orig.cpu(), recon.cpu()
    #     rms1 = ((torch.sum(orig ** 2) / N) ** 0.5)
    #     rms2 = ((torch.sum((orig - recon) ** 2) / N) ** 0.5)
    #     snr = 10 * torch.log10((rms1 / rms2) ** 2)
    #     return snr

    def test(self, mode='test'):
        self.eval_mode()

        with torch.no_grad():
            avg_carrier_loss, avg_msg_loss,total_loss = 0, 0, 0
            carrier_snr_list = []
            msg_snr_list = []

            logger.info(f"phase: {'test' if mode == 'test' else 'validation'}")
            loader = self.test_loader if mode == 'test' else self.val_loader
            # start of training loop
            logger.info("start testing...")
            for idx, train_batch in enumerate(tqdm(loader)):
                data = train_batch.float()

                # Saves secret images and secret covers
                covers = data[:len(data) // 2]
                secrets = [data[len(data) // 2:]] # is a list
                # feedforward and incur loss
                carrier_reconst, msg_reconst = self.forward(covers, secrets)
                loss, losses_log = self.incur_loss(covers, carrier_reconst, secrets, msg_reconst)
                avg_carrier_loss += losses_log['carrier_loss']
                avg_msg_loss += losses_log['avg_msg_loss']
                total_loss += loss

                # # calculate SnR for msg
                # msg_snr = 0
                # for m_spect, m_reconst in zip(msg, msg_reconst):
                #     msg_snr += self.snr(m_spect, m_reconst)
                # msg_snr_list.append(msg_snr / self.n_messages)
                #
                # # calculate SnR for carrier
                # carrier_snr = self.snr(carrier, carrier_reconst)
                # carrier_snr_list.append(carrier_snr)

            logger.info("finished testing!")
            logger.info(f"carrier loss: {avg_carrier_loss/idx+1}")
            logger.info(f"total loss: {total_loss/idx+1}")
            logger.info(f"message loss: {avg_msg_loss/idx+1}")
            # logger.info(f"message SnR: {np.mean(msg_snr_list)}")

        return {'{}/ epoch carrier loss'.format(mode): avg_carrier_loss/idx+1,
                '{}/ epoch msg loss'.format(mode): avg_msg_loss/idx+1,
                '{}/ epoch total loss'.format(mode): total_loss/idx+1,
                }

    # def sample_examples(self, n_examples=1, subdir=None):
    #     if self.mode != 'test':
    #         logger.warning("generating audio not in test mode!")
    #
    #     examples_dir = self.samples_dir
    #     if subdir is not None:
    #         examples_dir = join(examples_dir, subdir)
    #     makedirs(examples_dir, exist_ok=True)
    #
    #     logger.debug(f"generating {n_examples} examples in '{subdir}'")
    #     for i in range(n_examples):
    #         examples_subdir = join(examples_dir, f'{i}')
    #         makedirs(examples_subdir, exist_ok=True)
    #         carrier_path, msg_path = self.val_loader.dataset.spect_pairs[i]
    #         convert(self,
    #                 carrier_path,
    #                 msg_path,
    #                 trg_dir=examples_subdir,
    #                 epoch=i,
    #                 trim_start=self.trim_start,
    #                 num_samples=self.num_samples)
    #     logger.debug("done")