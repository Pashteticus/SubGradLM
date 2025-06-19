from datasets import load_dataset
from utils.parsers_utils import (
    reasonings_filename,
    save_dataset,
    format_big_russian,
    DsType,
    format_distill,
)

FILENAME = reasonings_filename()
SIZE = 20 * 1024 * 1024 * 1024
index = 0
with load_dataset("ZeroAgency/ru-big-russian-dataset").filter(
    lambda row: row["has_reasoning"]
) as ds:
    index = save_dataset(
        ds,
        DsType.REASONING,
        FILENAME,
        format_big_russian,
        max_file_size=SIZE / 2,
        batch_size=50000,
        need_header=True,
    )
with load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1") as ds:
    save_dataset(
        ds,
        DsType.REASONING,
        FILENAME,
        format_distill,
        index=index,
        max_file_size=SIZE,
        batch_size=50000,
    )