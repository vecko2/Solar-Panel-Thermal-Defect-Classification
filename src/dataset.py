import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
try:
    from torch.utils.data import Dataset
except (ImportError, OSError):
    Dataset = object


CLASSES = (
    "Cell", "Cell-Multi", "Cracking", "Diode", "Diode-Multi", "Hot-Spot",
    "Hot-Spot-Multi", "No-Anomaly", "Offline-Module", "Shadowing", "Soiling", "Vegetation",
)
LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
SEED = 42
IMAGE_SIZE = (24, 40)


def load_metadata(data_dir="data", validate_images=True):
    data_dir = Path(data_dir).resolve()
    metadata_path = data_dir / "module_metadata.json"
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Metadata must be a nonempty object keyed by image ID")

    def read_record(item):
        image_id, record = item
        if not isinstance(record, dict):
            return None, f"{image_id}: invalid metadata record"
        relative_path = record.get("image_filepath")
        label = record.get("anomaly_class")
        if not isinstance(relative_path, str) or not relative_path.startswith("images/"):
            return None, f"{image_id}: invalid image path"
        path = (data_dir / relative_path).resolve()
        if not path.is_relative_to(data_dir / "images") or path.suffix.lower() != ".jpg":
            return None, f"{image_id}: unsafe image path"
        if label not in LABEL_TO_INDEX:
            return None, f"{image_id}: unknown class {label!r}"
        row = {"image_id": str(image_id), "image_filepath": relative_path,
               "anomaly_class": label, "label": LABEL_TO_INDEX[label]}
        if validate_images:
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.size != IMAGE_SIZE or image.mode != "L":
                        raise ValueError(f"expected grayscale {IMAGE_SIZE}, got {image.mode} {image.size}")
                    row["pixel_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
                    pixels = list(image.resize((9, 8)).getdata())
                    bits = sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
                               for y in range(8) for x in range(8))
                    row["dhash"] = f"{bits:016x}"
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                return None, f"{image_id}: {exc}"
        return row, None

    with ThreadPoolExecutor(max_workers=16) as pool:
        checked = pool.map(read_record, sorted(metadata.items()))
        results = list(checked)
    rows = [row for row, error in results if error is None]
    errors = [error for row, error in results if error is not None]
    if errors:
        raise ValueError(f"Dataset validation failed ({len(errors)} records):\n" + "\n".join(errors[:20]))
    frame = pd.DataFrame(rows)
    if set(frame.anomaly_class) != set(CLASSES):
        raise ValueError("Metadata does not contain all 12 expected classes")
    if frame.image_filepath.duplicated().any():
        raise ValueError("Multiple image IDs point to the same file")
    return frame


def make_splits(frame, seed=SEED):
    if "pixel_sha256" not in frame:
        raise ValueError("Validate images before splitting")
    conflicts = frame.groupby("pixel_sha256").anomaly_class.nunique()
    excluded_hashes = set(conflicts.index[conflicts > 1])
    groups = frame.loc[~frame.pixel_sha256.isin(excluded_hashes)].drop_duplicates("pixel_sha256")[["pixel_sha256", "anomaly_class"]]
    train, holdout = train_test_split(groups, test_size=0.30, stratify=groups.anomaly_class, random_state=seed)
    val, test = train_test_split(holdout, test_size=0.50, stratify=holdout.anomaly_class, random_state=seed)
    split_by_hash = {key: name for name, part in (("train", train), ("val", val), ("test", test))
                     for key in part.pixel_sha256}
    manifest = frame.copy()
    manifest["split"] = manifest.pixel_sha256.map(split_by_hash).fillna("excluded")
    if manifest.image_id.duplicated().any() or manifest.split.isna().any():
        raise ValueError("Invalid split assignment")
    return manifest.sort_values("image_id").reset_index(drop=True)


def save_manifest(manifest, path="results/metrics/split_manifest.csv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return path


def get_transform(train=False):
    from torchvision import transforms

    steps = [transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose([*steps, transforms.ToTensor()])


class SolarThermalDataset(Dataset):
    def __init__(self, manifest, split, data_dir="data", transform=None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.rows = manifest.loc[manifest.split.eq(split)].reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform or get_transform(train=split == "train")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.data_dir / row.image_filepath) as image:
            return self.transform(image.convert("L")), int(row.label)
