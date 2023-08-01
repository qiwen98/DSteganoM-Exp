import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from os.path import join
from os import makedirs
from CNNModel import Encoder, CarrierDecoder, MsgDecoder
from Utils import corrupt
from solver import Solver

## this will In

class SolverStegano(Solver):
    def __init__(self, config):
        super(SolverStegano, self).__init__(config)
        logger.info("running multiple decoders solver!")

        # ------ create models ------
        #self.dec_c_conv_dim = self.n_messages + 1 + 64
        self.build_models()

        # ------ make parallel ------
        # self.enc_c = nn.DataParallel(self.enc_c)
        # self.dec_c = nn.DataParallel(self.dec_c)
        # self.dec_m = [nn.DataParallel(m) for m in self.dec_m]


        # ------ create optimizer ------
        dec_m_params = []
        for i in range(len(self.dec_m)):
            dec_m_params += list(self.dec_m[i].parameters())
        params = list(self.enc_c.parameters()) \
               + list(self.dec_c.parameters()) \
               + list(dec_m_params)
        self.opt = self.opt_type(params, lr=self.lr)
        if self.lr_decay:
            self.lr_sched = StepLR(self.opt, step_size=20, gamma=0.5)

        # ------ send to cuda ------
        self.enc_c.to(self.device)
        self.dec_c.to(self.device)
        self.dec_m = [m.to(self.device) for m in self.dec_m]

        if self.load_ckpt_dir != None:
            self.load_models(self.load_ckpt_dir)

        logger.debug(self.enc_c)
        logger.debug(self.dec_c)
        logger.debug(self.dec_m)

    def build_models(self):
        # super(SolverStegano, self).build_models() ##no need this line because nothing to call in slove.py

        self.enc_c = Encoder(block_type=self.block_type,
                             n_layers=self.config.enc_n_layers)

        self.dec_c = CarrierDecoder(conv_dim=self.dec_c_conv_dim,
                                    block_type=self.block_type,
                                    n_layers=self.config.dec_c_n_layers)

        self.dec_m = [MsgDecoder(conv_dim=self.dec_m_conv_dim,
                                 block_type=self.block_type) for _ in range(self.n_messages)]

    def save_models(self, suffix=''):
        logger.info(f"saving model to: {self.ckpt_dir}\n==> suffix: {suffix}")
        makedirs(self.ckpt_dir, exist_ok=True)
        torch.save(self.enc_c.state_dict(), join(self.ckpt_dir,"{}_enc_c.pt".format(suffix)))
        torch.save(self.dec_c.state_dict(), join(self.ckpt_dir,"{}_dec_c.pt".format(suffix)))
        for i,m in enumerate(self.dec_m):
            torch.save(m.state_dict(), join(self.ckpt_dir, "{}_dec_m_{}.pt".format(suffix,i)))

    def load_models(self, ckpt_dir):
        run = self.experiment.get_summary()
        suffix = "epoch{0}_{1}_".format(run.summary['lowest_cost_epoch'],self.block_type)

        self.enc_c.load_state_dict(torch.load(join(ckpt_dir, suffix, "enc_c.pt")))
        self.dec_c.load_state_dict(torch.load(join(ckpt_dir, suffix, "dec_c.pt")))
        for i,m in enumerate(self.dec_m):
            m.load_state_dict(torch.load(join(ckpt_dir, f"dec_m_{i}.pt")))
        logger.info("loaded models")

    def reset_grad(self):
        self.opt.zero_grad()

    def train_mode(self):
        super(SolverStegano, self).train_mode()
        self.enc_c.train()
        self.dec_c.train()
        for model in self.dec_m:
            model.train()

    def eval_mode(self):
        super(SolverStegano, self).eval_mode()
        self.enc_c.eval()
        self.dec_c.eval()
        for model in self.dec_m:
            model.eval()

    def step(self):
        self.opt.step()
        if self.cur_iter % len(self.train_loader) == 0:
            self.lr_sched.step()

    def incur_loss(self, carrier, carrier_reconst, msg, msg_reconst):
        n_messages = len(msg)
        losses_log = defaultdict(int)
        carrier, msg = carrier.to(self.device), [msg_i.to(self.device) for msg_i in msg]
        all_msg_loss = 0
        carrier_loss = self.reconstruction_loss(carrier_reconst, carrier, type=self.loss_type)
        for i in range(n_messages):
            msg_loss = self.reconstruction_loss(msg_reconst[i], msg[i], type=self.loss_type)
            all_msg_loss += msg_loss
        losses_log['carrier_loss'] = carrier_loss.item()
        losses_log['avg_msg_loss'] = all_msg_loss.item() / self.n_messages
        loss = carrier_loss + self.gamma_msg_loss * all_msg_loss

        return loss, losses_log

    def forward(self, carrier, msg):
        # logger.debug("carrier type{}".format(type(carrier)))
        # logger.debug("msg type{}".format(type(msg)))
        # assert type(carrier) == torch.Tensor and type(msg) == list

        msg_reconst_list = []
        N, C, V, T = carrier.size()

        carrier,  msg = carrier.to(self.device), \
                                      [msg_i.to(self.device) for msg_i in msg]


        # create embedded carrier
        carrier_enc = self.enc_c(carrier)  # encode the carrier
        msg_enc = torch.cat(msg, dim=1)  # concat all msg_i into single tensor
        merged_enc = torch.cat((carrier_enc, carrier, msg_enc), dim=1)  # concat encodings on features axis
        carrier_reconst = self.dec_c(merged_enc)  # decode carrier


        ỹ_cover_corrupted = carrier_reconst.permute(0, 3, 2, 1).contiguous().view(N * T, V, C)
        # print("ỹ_cover_corrupted",ỹ_cover_corrupted.shape)
        # do corruption
        ỹ_cover_corrupted = corrupt(ỹ_cover_corrupted, self.sigma_o, self.sigma_s, self.beta)
        # decoder
        # recover from corrupt shape   #N,T,V,C -> N,T,V,T
        x = ỹ_cover_corrupted.unsqueeze(0).permute(0, 3, 2, 1).contiguous()
        # logger.debug(x.shape)
        # decode messages from carrier
        for i in range(len(msg)):  # decode each msg_i using decoder_m_i
            msg_reconst = self.dec_m[i](x)
            msg_reconst_list.append(msg_reconst)

        return carrier_reconst, msg_reconst_list