from datasets import load_dataset
from src.utils.parsers_utils import *

FILENAME = "Reasonings.csv"
SIZE =20 * 1024 * 1024 * 1024
index = 0
with load_dataset("ZeroAgency/ru-big-russian-dataset").filter(lambda row: row["has_reasoning"] == True) as ds:
  index = save_dataset(
    ds, DsType.REASONING, FILENAME, format_big_russian, max_file_size=SIZE / 2, batch_size=50000, need_header=True
    )
with load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1") as ds:
  save_dataset(
    ds,
    DsType.REASONING,
    FILENAME,
    format_distill,
    index=index,
    max_file_size=SIZE,
    batch_size=50000
)
