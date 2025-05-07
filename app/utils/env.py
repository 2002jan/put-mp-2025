import os
import pickle
from pathlib import Path

from dotenv import dotenv_values


class Env:

    def __init__(self):
        file_path = Path(os.path.dirname(os.path.realpath(__file__)))
        project_path = file_path.parent.parent.absolute()
        env_path = os.path.join(project_path, '.env')

        env_values = dotenv_values(env_path)

        # Add defaults here
        config_values = {
            "DATASET_PATH": os.path.join(project_path, 'data'),
            "CACHE_ZIPS": False,
            "DATASET_SIZE": 'full',
            "BOTH_CAMERAS": False,
            **env_values
        }

        dataset_types = self._load_dataset_types()

        # Add envs here
        self.dataset_path = Path(config_values["DATASET_PATH"])
        self.cache_zips = config_values["CACHE_ZIPS"].lower() in ['true', '1']
        dataset_size = config_values["DATASET_SIZE"] if config_values["DATASET_SIZE"] in dataset_types.keys() else 'full'
        self.dataset_drives = dataset_types[dataset_size]
        self.both_cameras = config_values["BOTH_CAMERAS"].lower() in ['true', '1']


    def _load_dataset_types(self):
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        path = path.parent.absolute()
        path = path / 'data' / 'dataset_types.pickle'

        with open(path, 'rb') as f:
            return pickle.load(f)
