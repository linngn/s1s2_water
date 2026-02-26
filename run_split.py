"""Run split_train_dataset with timestamped output directory."""
from datetime import datetime
from pathlib import Path
from prepare.split import split_train_dataset

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path(r"D:\waterbody_data_prepare\train_dataset") / timestamp

split_train_dataset(
    data_dir=r"D:\s1s2_water\data",
    out_dir=str(out_dir),
    patch_shape=(256, 256),
    val_ratio=0.3,
    seed=42,
)
