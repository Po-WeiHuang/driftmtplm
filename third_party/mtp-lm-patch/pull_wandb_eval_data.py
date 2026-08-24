import time
import pandas as pd
import wandb
from tqdm import tqdm

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("po-wei-huang-university-of-oxford/singleshot-evals")

summary_list, config_list, name_list, tag_list = [], [], [], []
for run in tqdm(runs, desc="Pulling wandb eval data"):
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

    # add the tags as well
    tag_list.append(run.tags)

runs_df = pd.DataFrame(
    {
        "summary": summary_list,
        "config": config_list,
        "name": name_list,
        "tags": tag_list,
    }
)

timestamp = time.strftime("%Y%m%d-%H%M%S")
runs_df.to_csv(f"figures_data_raw/singleshot-evals_summary_table_{timestamp}.csv")
