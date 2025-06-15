from utils.parsers_utils import (
    save_dataset,
    DsType,
    format_evol,
    format_everything,
    format_open_orca,
)
from datasets import load_dataset

FILENAME = "Instructions.csv"
ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")
save_dataset(ds, DsType.INSTRUCT, FILENAME, format_evol, need_header=True)
ds = load_dataset("rombodawg/Everything_Instruct")
save_dataset(ds, DsType.INSTRUCT, FILENAME, format_everything)
ds = load_dataset("d0rj/OpenOrca-ru")
save_dataset(ds, DsType.INSTRUCT, FILENAME, format_open_orca)
