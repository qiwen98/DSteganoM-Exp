
import wandb
api = wandb.Api()

def get_historical_run(run_id: str):
    """Allows restoring an historical run to a writable state
    """
    return wandb.init(id=wandb.Api().run(run_id).id, resume='allow')

run = get_historical_run("stegano-m/stegano_M_valid_project/2bfs6h72")
print(run)
MODEL_ARTIFACT_NAME = 'run-2bfs6h72-best_epoch_10_Cost'
model_artifact = wandb.Artifact(MODEL_ARTIFACT_NAME, type='run_table')
table = wandb.Table(columns=["a", "b", "c"], data=[[1, 1*2, 2**1]])

model_artifact.add(table,MODEL_ARTIFACT_NAME)
run.log_artifact(model_artifact)


# run = api.run("stegano-m/stegano_M_valid_project/2bfs6h72") #this id need change
# artifact = api.artifact('stegano-m/stegano_M_valid_project/run-2bfs6h72-Test10_Cost:v0', type='run_table')
#
# # artifact = wandb.Artifact('my_table', 'dataset')
# table = wandb.Table(columns=["a", "b", "c"], data=[[1, 1*2, 2**1]])
#
# artifact["my_table"] = table

# artifact.save