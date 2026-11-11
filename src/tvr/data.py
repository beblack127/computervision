"""MSR-VTT dataset/dataloaders: official X-Pool logic + optional frame cache.

Differences from third_party/xpool/datasets/msrvtt_dataset.py:
  * annotation paths are absolute (official code assumes cwd == repo root);
  * frames come from tvr.frame_cache when --cache_dir is set (bit-identical, faster);
  * transforms come from tvr.compat (antialias pinned to torchvision-0.9 behaviour).
Caption pairing, ordering, split files and DataLoader settings are unchanged.
"""
import os

import torch
from torch.utils.data import DataLoader

from tvr.compat import init_transform_dict
from tvr.frame_cache import load_frames
from tvr.xpool_bridge import ensure_xpool_on_path

ensure_xpool_on_path()
from datasets.msrvtt_dataset import MSRVTTDataset  # noqa: E402  (official)
from datasets.video_capture import VideoCapture  # noqa: E402  (official)
from modules.basic_utils import load_json  # noqa: E402
import pandas as pd  # noqa: E402


class ProjectMSRVTTDataset(MSRVTTDataset):
    def __init__(self, config, split_type="train", img_transforms=None, frame_order="original"):
        # NOTE: we do not call super().__init__ because it hard-codes relative paths.
        self.config = config
        self.videos_dir = config.videos_dir
        self.cache_dir = getattr(config, "cache_dir", None)
        self.img_transforms = img_transforms
        self.split_type = split_type
        self.frame_order = frame_order  # analysis only: original | reversed | shuffled | middle
        data_dir = config.data_dir
        db_file = os.path.join(data_dir, "MSRVTT_data.json")
        test_csv = os.path.join(data_dir, "MSRVTT_JSFUSION_test.csv")
        train_csv = os.path.join(data_dir, "MSRVTT_train.7k.csv" if config.msrvtt_train_file == "7k"
                                 else "MSRVTT_train.9k.csv")
        self.db = load_json(db_file)
        if split_type == "train":
            train_df = pd.read_csv(train_csv)
            self.train_vids = train_df["video_id"].unique()
            self._compute_vid2caption()
            self._construct_all_train_pairs()
        else:
            self.test_df = pd.read_csv(test_csv)

    def _load_video(self, video_path, video_id):
        if self.cache_dir:
            arr = load_frames(self.cache_dir, video_id)
            imgs = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255
        else:
            imgs, _ = VideoCapture.load_frames_from_video(video_path, self.config.num_frames,
                                                          self.config.video_sample_type)
        return imgs

    def _reorder(self, imgs, index):
        if self.frame_order == "original":
            return imgs
        if self.frame_order == "reversed":
            return imgs.flip(0)
        if self.frame_order == "shuffled":
            g = torch.Generator().manual_seed(1000 + index)  # deterministic per sample
            return imgs[torch.randperm(imgs.shape[0], generator=g)]
        if self.frame_order == "middle":  # single middle frame repeated: no temporal info at all
            mid = imgs.shape[0] // 2
            return imgs[mid:mid + 1].repeat(imgs.shape[0], 1, 1, 1)
        raise ValueError(self.frame_order)

    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
        imgs = self._load_video(video_path, video_id)
        imgs = self._reorder(imgs, index)
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)
        return {"video_id": video_id, "video": imgs, "text": caption}


def get_data_loader(config, split_type="train", frame_order="original", shuffle=None):
    tf = init_transform_dict(config.input_res)
    if config.dataset_name != "MSRVTT":
        raise NotImplementedError("Only MSR-VTT is wired up in this project.")
    if split_type == "train":
        ds = ProjectMSRVTTDataset(config, "train", tf["clip_train"], frame_order)
        return DataLoader(ds, batch_size=config.batch_size, shuffle=True if shuffle is None else shuffle,
                          num_workers=config.num_workers, pin_memory=True)
    ds = ProjectMSRVTTDataset(config, "test", tf["clip_test"], frame_order)
    # persistent_workers is left off on purpose: it would change how worker RNGs and the
    # shuffle RNG are consumed across epochs relative to the official code.
    return DataLoader(ds, batch_size=config.batch_size, shuffle=False,
                      num_workers=config.num_workers, pin_memory=True)
