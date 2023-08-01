from torch_geometric_temporal.dataset import MTMDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch
from CNNModel_V2 import RecurrentGCN
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from Utils import customized_loss
from functools import lru_cache
from torch.utils.data.sampler import SubsetRandomSampler
from Utils import H5_Dataset,DataSplit,Custom_Dataset
from loguru import logger
from torch.optim.lr_scheduler import StepLR
import glob
import wandb
from os.path import join
from torchvision import transforms
from os import makedirs
torch.manual_seed(0)
import gc

import argparse



parser = argparse.ArgumentParser(description="testing and prepare npy for plot")
parser.add_argument("--dataset", type=str, default="Combinedataset", help="Dataset, default is MTM.")
parser.add_argument("--isCorrupted", type=str, default="uncorrupted", help="corrupted or not, uncorrupted/corrupted")
parser.add_argument("--module", type=str, default="normal", help="Dataset, default is normal")
parser.add_argument("--lossBeta", type=float, default=2, help="the beta weight in loss to control secret loss")
parser.add_argument("--ENC_C_N_LAYERS", type=int, default=3, help="the beta weight in loss to control secret loss")
parser.add_argument("--DEC_C_N_LAYERS", type=int, default=4, help="the beta weight in loss to control secret loss")
parser.add_argument("--DEC_M_N_LAYERS", type=int, default=6, help="the beta weight in loss to control secret loss")



args_opt = parser.parse_args()



CONFIG = {"dataset":"MTMdataset",
          "type":"variant-3",
          "epochs": 100,
          "learning-rate": 0.001,
          "module":"{}Block_{}_layer_M_Dec".format(args_opt.module,args_opt.DEC_M_N_LAYERS)
          }




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def extract_model_state_dict(ckpt_path, model_name='model', ):
    checkpoint = torch.load(ckpt_path, map_location='cuda:0')
    checkpoint_ = {}
    prefixes_to_ignore = []
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        # print(k)
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', is_ERQcameraType=False):
    if not ckpt_path:
        return

    else:
        model_dict = model.state_dict()
        checkpoint_ = extract_model_state_dict(ckpt_path, model_name)
        model_dict.update(checkpoint_)
        model.load_state_dict(model_dict,strict=False)


if __name__ == "__main__":

    # ##wandb configure
    # wandb.login(key="49495d5fe2572603aa9d6cac0c2be52f1dcd1829")
    # wandb.init(name=CONFIG["module"],
    #            project='stegano_M_resubmition_project',
    #            config=CONFIG,
    #            notes='MTM resubmission project',
    #            tags=[args_opt.dataset,args_opt.isCorrupted,args_opt.module],
    #            save_code=False)

    # code = wandb.Artifact('project-source', type='code')
    # for path in glob.glob('**/*.py', recursive=True):
    #     print(path)
    #     code.add_file(path)
    # wandb.run.use_artifact(code)
    # ##wandb configure end

    #####################begin########################
    loader = MTMDatasetLoader()

    MTMdataset = loader.get_dataset(frames=24)


    def filter_OutZero(data):
        data = [i for i in data if i.sum() != 0]
        return data


    filtered_MTM_dataset = filter_OutZero(MTMdataset.features)



    custom_transforms = transforms.Compose([
        # transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        # transforms.Normalize(mean=[0.0136, -1.4150, 0.2607], std=[3.5221, 6.4809, 3.5314]),
        # transforms.Normalize(mean=[ 0.0122, -0.0583, 0.0244],std=[0.1738, 0.3198, 0.1742]),
        transforms.Normalize(mean=[-0.0642, -1.9103, 0.3576], std=[3.0516, 6.8471, 3.0204])  # new

    ])

    # Using the dataset object
    Combined_dataset = Custom_Dataset('CMU_Mocap.hdf5',MTMdataset.features,transform =custom_transforms)





    split = DataSplit(Combined_dataset, shuffle=True)
    print(split.get_train_split_point())
    print(split.get_validation_split_point())
    train_loader, val_loader, test_loader = split.get_split(batch_size=1, num_workers=0  )

    from tqdm import tqdm

    if args_opt.isCorrupted == 'uncorrupted':
        IS_CORRUPTED = False
    else:
        IS_CORRUPTED = True

    #beta for loss
    beta =  args_opt.lossBeta

    torch.cuda.empty_cache()
    model = RecurrentGCN(args_opt.module, args_opt.ENC_C_N_LAYERS, args_opt.DEC_C_N_LAYERS,args_opt.DEC_M_N_LAYERS,IS_CORRUPTED,dataset="Combined")
    model = model.to(device)
    load_ckpt(model,"ckpt\\uncorrupted\\Balujadataset\\Baluja_image_BalujaBlock_6_layer_M_Dec\\epoch8_Baluja_image_BalujaBlock_6_layer_M_Dec_state_dict_model.pt")
    scaler = torch.cuda.amp.GradScaler()
    # summary(model, ( 3, 24, 21))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    

    torch.manual_seed(0)


    def run_one_epoch(data_loader, experiment_mode, epoch):

        test_cost = 0

        for idx, (cover, secret) in enumerate(tqdm(data_loader)):
            with torch.cuda.amp.autocast():
                train_covers = cover.float()
                train_secrets = secret.float()
                ori_x = train_covers  # N,C,V,T
                ori_x_secret = train_secrets
                # print(ori_x.shape)
                # print(ori_x_secret.shape)

                # N,T,V,C = 1,24,21,3
                # x_cover_corrupted = ori_x.permute(0, 2, 3, 1).contiguous().view(N * T, V, C)
                # x_cover_corrupted = corrupt(x_cover_corrupted ,SIGMA_O, SIGMA_S, BETA)
                # print(x_cover_corrupted.shape)
                x = ori_x.contiguous().cuda()  # N,T,V,C -> #N,C,T,V
                # print(x.shape)

                x_secret = ori_x_secret.contiguous().cuda()

                optimizer.zero_grad()
                carrier_reconst, msg_reconst = model(x, x_secret)
                # train_hidden, train_output = model(x_secret, x)
                average_loss, cover_loss, secret_loss = customized_loss(msg_reconst, carrier_reconst, x_secret, x, beta)
                # train_loss_cover, train_loss_secret = 0, 0
                # train_loss = torch.mean((train_hidden - ori_x) ** 2)
            if experiment_mode == "Train":
                scaler.scale(average_loss).backward()
                #average_loss.backward()
                scaler.step(optimizer)
                #optimizer.step()
                scaler.update()



            elif experiment_mode == "Test":
                test_cost = test_cost + average_loss

            #   epoch average loss
        print(' Epoch [{0}], {4}_Average_loss: {1:.5f}, : {4}_loss-cover{2:.5f} :{4}_loss_secret{3:.5f}'.format(
            epoch, average_loss, cover_loss, secret_loss, experiment_mode))

        if experiment_mode == "Test":
            return idx, test_cost

        else:
            return average_loss, cover_loss, secret_loss


    ckpt_dir = join(".\ckpt",args_opt.isCorrupted,CONFIG['dataset'],CONFIG["module"])
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5,verbose=True)
    lowest_cost = 100
    train_cost = 0
    train_cover_cost = 0
    train_secret_cost = 0
    #######
    val_cost = 0
    val_cover_cost = 0
    val_secret_cost = 0
    # Run training
    for epoch in tqdm(range(1, CONFIG["epochs"]+1)):
        model.train()
        train_cost, train_cover_cost, train_secret_cost = run_one_epoch(train_loader, experiment_mode="Train",
                                                                        epoch=epoch)
        # wandb configure
        wandb.log({
            "Train/ Epoch": epoch,
            "Train/ Loss total": train_cost,
            "Train/ Loss Cover": train_cover_cost,
            "Train/ Loss Secret": train_secret_cost,
        })
        # if train_cost < lowest_cost:
        #     wandb.run.summary["lowest_cost"] = train_cost
        #     wandb.run.summary["lowest_cost_epoch"] = epoch
        #     lowest_cost = train_cost
        #     makedirs(ckpt_dir, exist_ok=True)
        #     torch.save(model.state_dict(),
        #                join(ckpt_dir, "epoch{0}_{1}_state_dict_model.pt".format(epoch, CONFIG["module"])))

        if epoch % 2 == 0:
            print("Start Val epoch...")
            with torch.no_grad():
                model.eval()
                val_cost, val_cover_cost, val_secret_cost = run_one_epoch(val_loader, experiment_mode="Validate",
                                                                          epoch=epoch)
                # wandb configure
                wandb.log({
                    "Val/ Epoch": epoch,
                    "Val/ Loss total": val_cost,
                    "Val/ Loss Cover": val_cover_cost,
                    "Val/ Loss Secret": val_secret_cost,
                })

                if val_cost < lowest_cost:
                    wandb.run.summary["lowest_cost"] = val_cost
                    wandb.run.summary["lowest_cost_epoch"] = epoch
                    lowest_cost = val_cost
                    makedirs(ckpt_dir, exist_ok=True)
                    torch.save(model.state_dict(),
                               join(ckpt_dir, "epoch{0}_{1}_state_dict_model.pt".format(epoch, CONFIG["module"])))

        scheduler.step()
    ##save one time at 100 epochs
    torch.save(model.state_dict(),
               join(ckpt_dir, "epoch{0}_{1}_state_dict_model.pt".format(epoch, CONFIG["module"])))


    #Run testing
    print("Start 10 Test epoch...")
    columns = []
    rows= []
    for i in range(10):
        torch.manual_seed(i)
        with torch.no_grad():
            model.eval()
            cost = 0
            index = 0
            index, cost = run_one_epoch(test_loader, experiment_mode="Test", epoch=i)
            cost = cost / (index + 1)
            cost = cost.item()
            print("MSE: {:.4f}".format(cost))
            rows.append(cost)
            columns.append("score_" + str(i))
    test_table = wandb.Table(columns=columns)
    test_table.add_data(*rows)
    # ✨ W&B: Log predictions table to wandb
    wandb.log({"Test/ 10_Cost": test_table})






