import os
from pathlib import Path
import glob
module_path = str(Path.cwd().parents[0])
import typer
from tad.dataset.reader import GeneralDataset
from tad.model_utils.model_def import train_embedder

module_path = str(Path.cwd())

def run_configs(configs_folder_name: str,
                dataset: GeneralDataset,
                dataset_trace: int = None,
                overwrite: bool = False,
                load_weights: bool = False,
                config_to_run: str = None):

    dataset_name = dataset.name
    print(f'[INFO]: Module_path = {module_path}')
    configs = glob.glob(os.path.join(module_path, configs_folder_name, '*.yaml'))
    print(f'Found {len(configs)} configs to run')
    for config_path in configs:
        # check if already trained this config
        config_name = os.path.basename(config_path)
        if dataset_trace is None:
            checkpoint = Path(module_path, '../../resources/model_checkpoints', dataset_name, config_name.split('.')[0])
        else:
            checkpoint = Path(module_path, '../../resources/model_checkpoints', dataset_name, str(dataset_trace), config_name.split('.')[0])
            
        if config_to_run is not None:
            if config_name != config_to_run:
                print(f'Skipping....: Trained Model on {dataset_name} already exist for configuration {config_name}. To retrain specify overwrite True or provide config name with --config-to-run param.')
                continue
            else:
                overwrite = True
        else:
            if os.path.exists(checkpoint):
                if not overwrite and not load_weights:
                    print(f'Skipping....: Trained Model on {dataset_name} already exist for configuration {config_name}. To retrain specify overwrite True.')
                    continue
        
            
            
        print(f'[INFO]: Training {config_name} on {dataset_name}')
        _ = train_embedder(module_path,
                           dataset,
                           dataset_trace=dataset_trace,
                           config_path=config_path,
                           load_weights=load_weights
                          )
        print(f'[INFO]: Finished training {config_name} on {dataset_name}')


def run_configs_trace(configs_folder_name: str,
                dataset_name: str,                
                overwrite: bool = False,
                load_weights: bool = False,
                config_to_run: str = None):

    dataset = GeneralDataset.__members__[dataset_name.upper()] # for consistency with all the other scripts
    trainset, _, l_ = dataset()
    if type(trainset) is list:
        for i in range(len(trainset)):
            dataset_trace = i
            run_configs(configs_folder_name=configs_folder_name,
                dataset=dataset,
                dataset_trace=i,
                overwrite= overwrite,
                load_weights=load_weights,
                config_to_run=config_to_run)
            
    else:
        run_configs(configs_folder_name=configs_folder_name,
                dataset=dataset,
                dataset_trace=None,
                overwrite= overwrite,
                load_weights=load_weights,
                config_to_run=config_to_run)
        
if __name__ == "__main__":
    typer.run(run_configs_trace)