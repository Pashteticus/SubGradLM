from utils.parsers_utils import save_dataset, DsType, format_distill, format_big_russian
from datasets import load_dataset

FILENAME = "Reasonings.csv"
ds = load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1")
save_dataset(ds, DsType.REASONING, FILENAME, format_distill, need_header=True)
ds = load_dataset("ZeroAgency/ru-big-russian-dataset").filter(
    lambda row: row["has_reasoning"] == True
)
save_dataset(ds, DsType.REASONING, FILENAME, format_big_russian)
