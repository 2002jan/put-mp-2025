import logging
import os.path
import re
import sys
from zipfile import ZipFile

import requests
import tqdm
from pathlib import Path

from utils.env import Env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])


def main(config: Env):
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

        interesting_file = re.compile(r"image_0([23]).*\.png") if config.both_cameras else re.compile(r"image_02.*\.png")
        file_list = list(filter(interesting_file.search, file_list))

        a = set()

        for file in tqdm.tqdm(file_list, desc=f"Extracting files from {data_depth_annotated_file_path.name}",
                              unit="files"):
            f_path = Path(file)

            image_name = f_path.parent.name
            drive_name = f_path.parent.parent.parent.parent.name
            dataset_type = f_path.parent.parent.parent.parent.parent.name

            if drive_name not in config.dataset_drives:
                continue

            folder_path = config.dataset_path / dataset_type / drive_name / image_name / "target"

            os.makedirs(folder_path, exist_ok=True)

            data = zf.read(file)

            with open(folder_path / f_path.name, "wb") as of:
                of.write(data)

    if not config.cache_zips:
        logger.info(f"Removing {data_depth_annotated_file_path}")
        os.remove(data_depth_annotated_file_path)

    get_raw_images(config, "train")
    get_raw_images(config, "val")


def get_raw_images(config: Env, folder: str):
    target_path = config.dataset_path / folder

    drive_regex = re.compile(r"[0-9]+_[0-9]+_[0-9]+_drive_[0-9]+")

    for f in target_path.iterdir():
        if f.is_file():
            continue

        drive_name = drive_regex.findall(f.name)[-1]

        if f"{drive_name}_sync" not in config.dataset_drives:
            continue

        url = f"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{drive_name}/{drive_name}_sync.zip"

        zip_file_path = download_file(url, config.dataset_path)

        interesting_file = re.compile(r"image_0([23])/data.*\.png") if config.both_cameras else re.compile(r"image_02/data.*\.png")

        drive_path = target_path / f

        if not drive_path.exists():
            os.makedirs(drive_path)

        with ZipFile(zip_file_path, "r") as zf:

            file_list = zf.namelist()
            file_list = list(filter(interesting_file.search, file_list))

            for file in tqdm.tqdm(file_list, desc=f"Extracting files from {drive_name}_sync.zip", unit="files"):
                f_path = Path(file)

                camera = f_path.parent.parent.name

                new_file_path = drive_path / camera / "raw"

                os.makedirs(new_file_path, exist_ok=True)

                data = zf.read(file)

                with open(new_file_path / f_path.name, "wb") as of:
                    of.write(data)

        if not config.cache_zips:
            logger.info(f"Removing {zip_file_path}")
            os.remove(zip_file_path)

        # https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0011/2011_09_26_drive_0011_sync.zip


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
    main(Env())
