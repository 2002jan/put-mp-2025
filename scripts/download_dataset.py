import logging
import os.path
import pickle
import re
import sys
from zipfile import ZipFile

import requests
import tqdm
from pathlib import Path

from dotenv import dotenv_values

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])


class DownloadDatasetConfig:
    def __init__(self):
        file_path = Path(os.path.dirname(os.path.realpath(__file__)))
        project_path = file_path.parent.absolute()
        env_path = os.path.join(project_path, '.env')

        env_values = dotenv_values(env_path)

        config_values = {
            "DATASET_PATH": os.path.join(project_path, 'data'),
            **env_values
        }

        self.dataset_path = Path(config_values["DATASET_PATH"])


def main(config: DownloadDatasetConfig):
    if not config.dataset_path.exists():
        config.dataset_path.mkdir()

    if len([f for f in config.dataset_path.iterdir() if not f.match('*.zip')]) > 0:
        logger.error("Dataset folder not empty, clear it before downloading")
        exit(1)

    logger.info("Downloading dataset")
    data_depth_annotated_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"

    data_depth_annotated_file_path = download_file(data_depth_annotated_url, config.dataset_path)

    logger.info("Download complete, unzipping...")

    with ZipFile(data_depth_annotated_file_path, "r") as zf:

        file_list = zf.namelist()

        for file in tqdm.tqdm(file_list, desc=f"Extracting files from {data_depth_annotated_file_path.name}", unit="files"):
            zf.extract(file, path=config.dataset_path)


def download_file(url: str, save_path: Path):
    r = requests.get(url, stream=True)
    filename = url.split("/")[-1]
    file_path = save_path / filename

    if file_path.exists():
        logger.info(f"{filename} already exists, skipping download")
        return file_path

    content_length = int(r.headers.get("Content-length")) / 10 ** 6

    bar = tqdm.tqdm(total=content_length, unit="Mb", desc=f"Downloading {filename}")

    try:
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(1024 / 10 ** 6)
    except (Exception, KeyboardInterrupt) as e:
        os.remove(file_path)
        logger.error(e)
        raise e
    finally:
        bar.close()

    return file_path


if __name__ == '__main__':
    main(DownloadDatasetConfig())
