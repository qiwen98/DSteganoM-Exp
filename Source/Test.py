import os

import torch
from torch_geometric_temporal.dataset import MTMDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from CNNModel_V2 import RecurrentGCN
from torch.utils.data import Dataset, DataLoader
from Utils import customized_loss,filter_OutZero,customized_PSNR,calculate_per_part_loss
import numpy as np
import wandb
import argparse
from os.path import join
from pathlib import Path


parser = argparse.ArgumentParser(description="testing and prepare npy for plot")
parser.add_argument("--dataset", type=str, default="MTM", help="Dataset, default is MTM.")
parser.add_argument("--isCorrupted", type=str, default="uncorrupted", help="corrupted or not, uncorrupted/corrupted")
parser.add_argument("--module", type=str, default="normal", help="Dataset, default is normal")
parser.add_argument("--apikey", type=str, default="21lzg3cz", help="wandb api key !!! important")
parser.add_argument("--sigma_o", type=float, default=0.1, help="0.1 by default")


args_opt = parser.parse_args()






def get_historical_run(run_id: str):
    """Allows restoring an historical run to a writable state
    """
    print(wandb.Api().run(run_id).id)
    return wandb.init(id=wandb.Api().run(run_id).id,project='stegano_M_valid_project', resume='allow')


if __name__ == "__main__":

    api = wandb.Api()
    run = api.run("stegano-m/stegano_M_valid_project/{}".format(args_opt.apikey)) #this id need change
    run2 = get_historical_run("stegano-m/stegano_M_valid_project/{}".format(args_opt.apikey))


    loader = MTMDatasetLoader()

    dataset = loader.get_dataset(frames=24)

    train_dataset, test_val_dataset = temporal_signal_split(dataset, train_ratio=0.7)
    test_dataset, val_dataset = temporal_signal_split(test_val_dataset, train_ratio=0.5)

    filter_train_dataset = filter_OutZero(train_dataset.features)
    filter_val_dataset = filter_OutZero(val_dataset.features)
    filter_test_dataset = filter_OutZero(test_dataset.features)

    filter_dataset = filter_OutZero(dataset.features)
    # filter_test_val_dataset = filter_OutZero(test_val_dataset.features)
    # test_val_loader = DataLoader(filter_test_val_dataset, batch_size=2,shuffle=True, num_workers=2,drop_last=True)

    train_loader = DataLoader(filter_train_dataset, batch_size=2, shuffle=True, num_workers=2,
                              drop_last=True)  # ,worker_init_fn=seed_worker,generator=g)
    val_loader = DataLoader(filter_val_dataset, batch_size=2, shuffle=True, num_workers=2,
                            drop_last=True)  # ,worker_init_fn=seed_worker,
    #     generator=g)
    test_loader = DataLoader(filter_test_dataset, batch_size=2, shuffle=True, num_workers=2,
                             drop_last=True)  # ,worker_init_fn=seed_worker,generator=g)

    print(len(test_loader))
    # train_datset_list = iter(train_loader)
    # data = next(train_datset_list)
    # print(data)

    BLOCK_TYPE = args_opt.module  # skip, relu, CBAMAttention, normal, Baluja # ours->skip (combine network)
    ENC_C_N_LAYERS = 3
    DEC_C_N_LAYERS = 4
    # beta for loss
    beta = 1.5
    SIGMA_O=args_opt.sigma_o
    SIGMA_S = 0.1
    BETA = 0.1
    # corrupted ?
    if args_opt.isCorrupted=='uncorrupted':
        IS_CORRUPTED = False
    else:
        IS_CORRUPTED = True

    model = RecurrentGCN(BLOCK_TYPE,ENC_C_N_LAYERS,DEC_C_N_LAYERS,IS_CORRUPTED,SIGMA_O)
    model = model.cuda()

    ckpt_dir = join(".\ckpt",args_opt.isCorrupted,run.config['dataset'],run.config["module"])
    run.summary["lowest_cost_epoch"]=100

    model.load_state_dict(torch.load(join(ckpt_dir,"epoch{0}_{1}_state_dict_model.pt".format(run.summary["lowest_cost_epoch"]
                                                                               ,run.config['module']))),strict=False)
    #model.load_state_dict(torch.load("epoch41_GatedBlock_state_dict_model.pt"))

    torch.manual_seed(0)

    def test():
        with torch.no_grad():
          model.eval()
          cost = 0
          PSNR = 0
          average_train_loss_for_plot=[]
          for idx, train_batch in enumerate(test_loader):
              data = train_batch.float()


              # Saves secret images and secret covers
              train_covers = data[:len(data) // 2]
              train_secrets = data[len(data) // 2:]
              ori_x = train_covers  # N,C,V,T
              ori_x_secret = train_secrets

              # N,T,V,C = 1,24,21,3
              # x_cover_corrupted = ori_x.permute(0, 2, 3, 1).contiguous().view(N * T, V, C)
              # x_cover_corrupted = corrupt(x_cover_corrupted ,SIGMA_O, SIGMA_S, BETA)
              # print(x_cover_corrupted.shape)
              x = ori_x.contiguous().cuda()  # N,T,V,C -> #N,C,T,V
              # print(x.shape)

              x_secret = ori_x_secret.contiguous().cuda()


              carrier_reconst, msg_reconst = model(x, x_secret)
              # train_hidden, train_output = model(x_secret, x)
              train_loss, train_loss_cover, train_loss_secret = customized_loss(msg_reconst, carrier_reconst, x_secret, x,
                                                                                beta)
              cost = cost + train_loss
              average_train_loss_for_plot.append(calculate_per_part_loss(msg_reconst, carrier_reconst, x_secret, x,
                                                                                1))

              # calculate PSNR per and add them together, we calculate the average at line 148
              PSNR_total_per_batch,PSNR_cover,PSNR_secret = customized_PSNR(msg_reconst, carrier_reconst, x_secret, x,
                                                                                1)

              PSNR = PSNR + PSNR_total_per_batch


        return idx, cost, PSNR,average_train_loss_for_plot


    with torch.no_grad():
        model.eval()
        cost = 0
        index = 0
        MSE = 0
        index, cost, PSNR,average_train_loss_for_plot = test()
        cost = cost / (index + 1)
        cost = cost.item()
        PSNR = PSNR / (index + 1)
        PSNR = PSNR.item()
        print("MSE: {:.4f}".format(cost))
        print("PSNR: {:.4f}".format(PSNR))


    final_plot_output=np.stack( average_train_loss_for_plot, axis=0 )


    test_file_name= 'PSNR_result_{}_{}_{}_{}.npy'.format(args_opt.isCorrupted,args_opt.dataset,args_opt.module,args_opt.sigma_o)

    dir=Path('./PSNR_results')
    os.makedirs(dir, exist_ok=True)
    test_file_dir = join(dir,test_file_name)
    # wrtie the test results
    with open( test_file_dir, 'wb') as f:
        np.save(f, PSNR)

    # # Run testing
    # print("Start 10 Test epoch...")
    # columns = []
    # rows = []
    # PSNR_columns = []
    # PSNR_rows = []
    # for i in range(10):
    #   torch.manual_seed(i)
    #   with torch.no_grad():
    #       model.eval()
    #       cost = 0
    #       index = 0
    #       MSE = 0
    #       index, cost, PSNR,average_train_loss_for_plot = test()
    #       cost = cost / (index + 1)
    #       cost = cost.item()
    #       PSNR = PSNR / (index + 1)
    #       PSNR = PSNR.item()
    #       print("MSE: {:.4f}".format(cost))
    #       print("PSNR: {:.4f}".format(PSNR))
    #       rows.append(cost)
    #       columns.append("score_" + str(i))
    #       PSNR_rows.append(PSNR)
    #       PSNR_columns.append("PSNR_" + str(i))
    # test_table = wandb.Table(columns=columns)
    # PSNR_table = wandb.Table(columns=PSNR_columns)
    # PSNR_table.add_data(*PSNR_rows)
    # test_table.add_data(*rows)
    # # ✨ W&B: Log predictions table to wandb
    # # MODEL_ARTIFACT_NAME = 'run-2bfs6h72-best_epoch_10_Cost'
    # # model_artifact = wandb.Artifact(MODEL_ARTIFACT_NAME, type='run_table')
    # # model_artifact.add(test_table, MODEL_ARTIFACT_NAME)
    # # run2.log_artifact(model_artifact)
    # wandb.log({"Test/ best_epoch_10_Cost": test_table})
    # wandb.log({"PSNR": PSNR_table})



