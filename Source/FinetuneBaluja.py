from torch_geometric_temporal.dataset import MTMDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch
from torch.optim.lr_scheduler import StepLR
from CNNModel_V2 import RecurrentGCN
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from Utils import customized_loss
from functools import lru_cache
from loguru import logger
import glob
import wandb
from os.path import join
from os import makedirs
torch.manual_seed(0)

import argparse



parser = argparse.ArgumentParser(description="testing and prepare npy for plot")
parser.add_argument("--dataset", type=str, default="MTMdataset", help="Dataset, default is MTM.")
parser.add_argument("--isCorrupted", type=str, default="uncorrupted", help="corrupted or not, uncorrupted/corrupted")
parser.add_argument("--module", type=str, default="Baluja", help="Dataset, default is normal")
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





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def filter_OutZero(data):
    data = [i for i in data if i.sum() != 0]
    return data


if __name__ == "__main__":

    #wandb configure
    wandb.login(key="49495d5fe2572603aa9d6cac0c2be52f1dcd1829")
    wandb.init(name=CONFIG["module"],
               project='stegano_M_resubmition_project',
               config=CONFIG,
               notes='MTM resubmission project',
               tags=[args_opt.dataset,args_opt.isCorrupted,args_opt.module],
               save_code=False)

    code = wandb.Artifact('project-source', type='code')
    for path in glob.glob('**/*.py', recursive=True):
        print(path)
        code.add_file(path)
    wandb.run.use_artifact(code)
    #wandb configure end

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

    train_loader = DataLoader(filter_train_dataset, batch_size=2, shuffle=True, num_workers=0,
                              drop_last=True)  # ,worker_init_fn=seed_worker,generator=g)
    val_loader = DataLoader(filter_val_dataset, batch_size=2, shuffle=True, num_workers=0,
                            drop_last=True)  # ,worker_init_fn=seed_worker,
    #     generator=g)
    test_loader = DataLoader(filter_test_dataset, batch_size=2, shuffle=True, num_workers=0,
                             drop_last=True)  # ,worker_init_fn=seed_worker,generator=g)

    # Investigating the dataset
    print("Dataset type: ", type(dataset))
    print("Index tensor of edges ", dataset.edge_index.transpose())

    print("Edge weight tensor ", dataset.edge_weight)
    print("Length of node feature tensors: ",
          len(dataset.features))  # Length of node feature tensors:  14453 means there is 14453 samples in this dataset
    print("List of node feature tensors: ", dataset.features[
        0].shape)  # each sample have the shape of (3, 21, 16) 3 channel-> x,y,z , 21 node points, with 16 time frame
    print("List of node label (target) tensors: ", dataset.targets[
        0].shape)  # each sample have the the target shape of (16, 6) 16 time frame and target label class per frame  (in this case all 16 times frame should have the same class)

    # 6classes

    # Grasp, Release, Move, Reach, Position plus a negative class for frames without graph signals (no hand present).

    total = 0
    for time, snapshot in enumerate(train_dataset):
        # print(time) # the time here is just an index,, not neccesary to be a time
        if time == 1:
            # print(snapshot.x.shape,snapshot.y.shape, snapshot.edge_index.shape, snapshot.edge_attr.shape)
            break

    print(type(snapshot.x))
    print(snapshot.x[0])
    numpy_x = snapshot.x.numpy()
    print(type(numpy_x))
    # print(numpy_x[0])

    print(snapshot.y)

    from tqdm import tqdm

    if args_opt.isCorrupted == 'uncorrupted':
        IS_CORRUPTED = False
    else:
        IS_CORRUPTED = True

    #beta for loss
    beta =  args_opt.lossBeta

    torch.cuda.empty_cache()
    model = RecurrentGCN(args_opt.module, args_opt.ENC_C_N_LAYERS, args_opt.DEC_C_N_LAYERS,args_opt.DEC_M_N_LAYERS,IS_CORRUPTED,dataset="MTM")
    model = model.to(device)

    load_ckpt(model,"ckpt\\uncorrupted\\Balujadataset\\Baluja_image_BalujaBlock_6_layer_M_Dec\\epoch8_Baluja_image_BalujaBlock_6_layer_M_Dec_state_dict_model.pt")

    # summary(model, ( 3, 24, 21))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()

    model.train()

    torch.manual_seed(0)


    def run_one_epoch(data_loader, experiment_mode, epoch):

        test_cost = 0

        for idx, train_batch in enumerate(tqdm(data_loader)):
            data = train_batch.float()
            # Saves secret images and secret covers
            covers = data[:len(data) // 2]
            secrets = data[len(data) // 2:]
            ori_x = covers  # N,C,V,T
            ori_x_secret = secrets

            # N,T,V,C = 1,24,21,3
            # x_cover_corrupted = ori_x.permute(0, 2, 3, 1).contiguous().view(N * T, V, C)
            # x_cover_corrupted = corrupt(x_cover_corrupted ,SIGMA_O, SIGMA_S, BETA)
            # print(x_cover_corrupted.shape)
            x = ori_x.contiguous().cuda()  # N,T,V,C -> #N,C,T,V
            # print(x.shape)

            x_secret = ori_x_secret.contiguous().cuda()

            optimizer.zero_grad()
            carrier_reconst, msg_reconst = model(x, x_secret)
            #print(carrier_reconst,msg_reconst)
            # train_hidden, train_output = model(x_secret, x)
            average_loss, cover_loss, secret_loss = customized_loss(msg_reconst, carrier_reconst, x_secret, x, beta)
            # train_loss_cover, train_loss_secret = 0, 0
            # train_loss = torch.mean((train_hidden - ori_x) ** 2)
            if experiment_mode == "Train":
                scaler.scale(average_loss).backward()
                # average_loss.backward()
                scaler.step(optimizer)
                # optimizer.step()
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
        train_cost, train_cover_cost, train_secret_cost = run_one_epoch(train_loader, experiment_mode="Train", epoch=epoch)
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
                # torch.save(model.state_dict(),
                #            join(ckpt_dir, "epoch{0}_{1}_state_dict_model.pt".format(epoch, CONFIG["module"])))

        scheduler.step()
    ##save one time at 100 epochs
    # torch.save(model.state_dict(),
    #            join(ckpt_dir, "epoch{0}_{1}_state_dict_model.pt".format(epoch, CONFIG["module"])))


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






