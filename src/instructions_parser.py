from datasets import load_dataset, DatasetDict
from utils.parsers_utils import * 


FILENAME = "Instructions.csv"
SIZE = 20 * 1024
with load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K") as ds:
  index = save_dataset(
    ds, DsType.INSTRUCT, FILENAME, format_evol, need_header=True, batch_size=100, max_file_size=SIZE / 3
)
with load_dataset("rombodawg/Everything_Instruct") as ds:
  index = save_dataset(
    ds,
    DsType.INSTRUCT,
    FILENAME,
    format_everything,
    index=index,
    max_file_size=2 * SIZE / 3,batch_size=100,
)
with load_dataset("d0rj/OpenOrca-ru") as ds:
  save_dataset(
    ds, DsType.INSTRUCT, FILENAME, format_open_orca, index=index, max_file_size=SIZE,batch_size=100,
)
