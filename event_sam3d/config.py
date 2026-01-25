import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

IS_CLUSTER = os.path.exists("/cluster")
IS_LOCAL = not IS_CLUSTER
PROJ_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJ_DIR if IS_CLUSTER else PROJ_DIR.parent
CKPT_DIR = WORKSPACE_DIR / "checkpoints"

DATA_DIR = (
    Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else WORKSPACE_DIR / "data"
)
EDATA_DIR = DATA_DIR
if (DATA_DIR / "event_dataset").exists():
    REPLICA_DIR = DATA_DIR / "event_dataset/Replica_final"
    DEVD_DIR = DATA_DIR / "event_dataset/DEVD" / "DAVIS_DEPTH_SLAM"
else:
    REPLICA_DIR = DATA_DIR / "Replica_final"
    DEVD_DIR = DATA_DIR / "DEVD" / "DAVIS_DEPTH_SLAM"
RGBE_DIR = EDATA_DIR / "eventsam/RGBE-SEG"
MVSEC_DIR = EDATA_DIR / "mvsec/hdf5"
RELATED_DIR = WORKSPACE_DIR / "related_work"
SAM3_DIR = RELATED_DIR / "segm/sam3"
SAM3D_DIR = RELATED_DIR / "rec/sam-3d-objects"

REPLICA_SCENES = [
    "office0",
    "office1",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
]
MVSEC_SCENES=[
    "indoor_flying1_data",
    "indoor_flying2_data",
    "indoor_flying3_data",
    "indoor_flying4_data",
    # "outdoor_day1_data",
    # "outdoor_day2_data",
]

DEVD_SCENES = [
    "mahjong1",
    "mountain1",
    "table1",
    "testbed1",
]
