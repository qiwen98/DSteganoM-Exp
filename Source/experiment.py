import argparse
import wandb
import os
from os.path import join
from loguru import logger
import glob


class Experiment(object):
    def __init__(self, config):




        self.exp_name = "{}{}".format(config.block_type,'BlockLog')
        self.dataset="{}{}".format(config.dataset,'dataset')

        logger.info("parsing experiment config to wandb")

        self.CONFIG = {"dataset": self.dataset,
                  "type": "baseline",
                  "epochs": config.num_iters,
                  "learning-rate": config.lr,
                  "module": config.block_type
                  }


        wandb.login(key="49495d5fe2572603aa9d6cac0c2be52f1dcd1829")


        wandb.init(name=self.exp_name,
                   project='stegano_M_project_test',
                   config=self.CONFIG,
                   notes='MTM first sucessful project',
                   tags=[self.dataset, 'Proper Run'],
                   save_code=False)

        code = wandb.Artifact('project-source', type='code')
        for path in glob.glob('**/*.py', recursive=True):
            print(path)
            code.add_file(path)
        wandb.run.use_artifact(code)
        ##wandb configure end

        logger.info("Successfully parsing experiment config to wandb")



    def log_metric(self, metrics_dict, step=None):
        # log all metrics using writers
        for k, v in metrics_dict.items():

            # log in wandb
            # if self.wandb_exp:
            wandb.log({k: v}, step=step)
            logger.info("{}:{}".format(k,v))
        logger.info("Logged experiment sumamry to wandb")

    def update_summary(self,summary_dict):
        #wandb.run.summary["lowest_cost"] = train_loss
         #   wandb.run.summary["lowest_cost_epoch"] = epoch
        for k, v in summary_dict.items():
            wandb.run.summary["{}".format(k)] = v

        logger.info("Updated experiment summary to wandb")

    def get_summary(self):
        api = wandb.Api()
        run_id= wandb.run.id
        run = api.run("stegano-m/stegano_M_project/{}".format(run_id))  # this id need change

        logger.info("Get experiment summary from wandb")
        return run




#if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('--lr', type=float, default=0.001, help='')
    # parser.add_argument('--lr_decay', type=bool, default=False, help='learn with learning decay?')
    # parser.add_argument('--num_iters', type=int, default=100, help='number of epochs')
    # parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'abs'], help='loss function used for training')
    # parser.add_argument('--dataset', type=str, default='MTM', help='select dataset', choices=['mtm', 'other'])
    # parser.add_argument('--block_type', type=str, default='normal', choices=['normal', 'skip', 'bn', 'in', 'relu'], help='type of block for encoder/decoder')
    # args = parser.parse_args()
    #
    # exp = Experiment(args)
    # exp.log_metric({'metrics/loss': 0.5})
    # exp.log_metric({'metrics/loss': 0.4, 'metrics/acc': 0.99})
    # exp.update_summary({'lowest_traning_cost': 0.4, 'lowest_training_cost_epoch': 0.99})
