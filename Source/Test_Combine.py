import torch
from torch_geometric_temporal.dataset import MTMDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from CNNModel_V2 import RecurrentGCN
from torch.utils.data import Dataset, DataLoader
from Utils import customized_loss,filter_OutZero,Custom_Dataset,customized_PSNR
import wandb,tqdm
from os.path import join
from torchvision import transforms
from Utils import H5_Dataset,DataSplit

import argparse
from os.path import join


parser = argparse.ArgumentParser(description="testing and prepare npy for plot")
parser.add_argument("--dataset", type=str, default="MTM", help="Dataset, default is MTM.")
parser.add_argument("--isCorrupted", type=str, default="uncorrupted", help="corrupted or not, uncorrupted/corrupted")
parser.add_argument("--module", type=str, default="normal", help="Dataset, default is normal")
parser.add_argument("--apikey", type=str, default="21lzg3cz", help="wandb api key !!! important")

args_opt = parser.parse_args()

SIGMA_O = 0.1
SIGMA_S = 0.1
BETA = 0.1
beta = 1

EDGE_INDEX = torch.tensor([[ 0 , 1],
 [ 0 , 5],
 [ 0 , 9],
 [ 0 , 17],
 [ 1 , 2],
 [ 2 , 3],
 [ 3 , 4],
 [ 6 , 5],
 [ 6 , 7],
 [ 7 , 8],
 [ 9 ,10],
 [10 ,11],
 [11 ,12],
 [13 ,14],
 [14 ,15],
 [15 ,16],
 [17 ,18],
 [18 ,19],
 [19 ,20]])
def get_historical_run(run_id: str):
    """Allows restoring an historical run to a writable state
    """
    print(wandb.Api().run(run_id).id)
    return wandb.init(id=wandb.Api().run(run_id).id,project='stegano_M_valid_project', resume='allow')


if __name__ == "__main__":

    api = wandb.Api()
    run = api.run("stegano-m/stegano_M_valid_project/{}".format(args_opt.apikey)) #this id need change
    run2 = get_historical_run("stegano-m/stegano_M_valid_project/{}".format(args_opt.apikey))
    # run2.name = "SkipGatedBlockLog"
    # run2.save()

    loader = MTMDatasetLoader()
    MTMdataset = loader.get_dataset(frames=24)


    def filter_OutZero(data):
        data = [i for i in data if i.sum() != 0]
        return data


    filtered_MTM_dataset = filter_OutZero(MTMdataset.features)

    custom_transforms = transforms.Compose([
        # transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        transforms.Normalize(mean=[0.0136, -1.4150, 0.2607], std=[3.5221, 6.4809, 3.5314]),
        # transforms.Normalize(mean=[ 0.0122, -0.0583, 0.0244],std=[0.1738, 0.3198, 0.1742]),

    ])

    # Using the dataset object
    Combined_dataset = Custom_Dataset('CMU_Mocap.hdf5', MTMdataset.features, transform=custom_transforms)

    split = DataSplit(Combined_dataset, shuffle=True)
    print(split.get_train_split_point())
    print(split.get_validation_split_point())
    train_loader, val_loader, test_loader = split.get_split(batch_size=1, num_workers=0)




    BLOCK_TYPE = args_opt.module  # skip, relu, CBAMAttention, normal, Baluja # ours->skip (combine network)
    ENC_C_N_LAYERS = 3
    DEC_C_N_LAYERS = 4
    # beta for loss
    beta = 1.5
    # corrupted ?
    if args_opt.isCorrupted == 'uncorrupted':
        IS_CORRUPTED = False
    else:
        IS_CORRUPTED = True

    model = RecurrentGCN(BLOCK_TYPE, ENC_C_N_LAYERS, DEC_C_N_LAYERS, IS_CORRUPTED)
    model = model.cuda()

    ckpt_dir = join(".\ckpt", args_opt.isCorrupted, run.config['dataset'], run.config["module"])
    run.summary["lowest_cost_epoch"] = 100

    model.load_state_dict(torch.load(join(ckpt_dir,"epoch{0}_{1}_state_dict_model.pt".format(run.summary["lowest_cost_epoch"]
                                                                               ,run.config['module']))))

    #model.load_state_dict(torch.load("epoch41_GatedBlock_state_dict_model.pt"))

    torch.manual_seed(0)

    def test():
        with torch.no_grad():
          model.eval()
          cost = 0
          PSNR = 0
          for idx, (cover, secret) in enumerate(test_loader):
              train_covers = cover.float()
              train_secrets = secret.float()
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
              # calculate PSNR per and add them together, we calculate the average at line 148
              PSNR_total_per_batch, PSNR_cover, PSNR_secret = customized_PSNR(msg_reconst, carrier_reconst, x_secret, x,
                                                                              beta)
              if not torch.isneginf(PSNR_total_per_batch):
                PSNR = PSNR + PSNR_total_per_batch

          return idx, cost, PSNR



    # Run testing
    print("Start 10 Test epoch...")
    columns = []
    rows = []
    PSNR_columns = []
    PSNR_rows = []
    for i in range(10):
        torch.manual_seed(i)
        with torch.no_grad():
            model.eval()
            index, cost,PSNR = test()
            cost = cost / (index + 1)
            cost = cost.item()
            PSNR = PSNR / (index + 1)
            PSNR = PSNR.item()
            print("MSE: {:.4f}".format(cost))
            print("PSNR: {:.4f}".format(PSNR))
            rows.append(cost)
            columns.append("score_" + str(i))
            PSNR_rows.append(PSNR)
            PSNR_columns.append("PSNR_" + str(i))
    test_table = wandb.Table(columns=columns)
    PSNR_table = wandb.Table(columns=PSNR_columns)
    PSNR_table.add_data(*PSNR_rows)
    test_table.add_data(*rows)
    # ✨ W&B: Log predictions table to wandb
    # MODEL_ARTIFACT_NAME = 'run-2bfs6h72-best_epoch_10_Cost'
    # model_artifact = wandb.Artifact(MODEL_ARTIFACT_NAME, type='run_table')
    # model_artifact.add(test_table, MODEL_ARTIFACT_NAME)
    # run2.log_artifact(model_artifact)
    wandb.log({"Test/ best_epoch_10_Cost": test_table})
    wandb.log({"PSNR": PSNR_table})



