from enum import Enum
import pandas as pd
import psutil
import gc
import re
import os
class Language(Enum):
    ENG = 1
    RU = 2


class DsType(Enum):
    REASONING = 1
    INSTRUCT = 2

def instructions_filename():
    return "Instructions.csv"

def reasonings_filename():
    return "Reasonings.csv"

def language_to_string(language):
    LANGUAGE_TO_STRING = {Language.ENG: "eng", Language.RU: "ru"}
    return LANGUAGE_TO_STRING[language]


def create_df(ds_type):
    if ds_type is DsType.REASONING:
        return pd.DataFrame(columns=["System", "User", "Reasoning", "Assistant", "Language"])
    else:
        return pd.DataFrame(columns=["System", "User", "Assistant", "Language"])


def order_instruction_df(df):
    df.insert(0, "System", df.pop("System"))
    df.insert(1, "User", df.pop("User"))
    df.insert(2, "Assistant", df.pop("Assistant"))


def order_reasoning_df(df):
    df.insert(0, "System", df.pop("System"))
    df.insert(1, "User", df.pop("User"))
    df.insert(2, "Reasoning", df.pop("Reasoning"))
    df.insert(3, "Assistant", df.pop("Assistant"))


def format_evol(df):
    df.rename(columns={"instruction": "User", "response": "Assistant"}, inplace=True)
    df["System"] = ""
    order_instruction_df(df)
    df["Language"] = language_to_string(Language.ENG)
    return df


def format_everything(df):
    df.rename(
        columns={"instruction": "System", "input": "User", "output": "Assistant"},
        inplace=True,
    )
    order_instruction_df(df)
    df["Language"] = language_to_string(Language.ENG)
    return df


def format_open_orca(df):
    df.drop(columns=["id"], inplace=True)
    df.rename(
        columns={
            "system_prompt": "System",
            "question": "User",
            "response": "Assistant",
        },
        inplace=True,
    )
    order_instruction_df(df)
    df["Language"] = language_to_string(Language.RU)
    return df


def format_distill(df):
    df["System"] = ""
    df["User"] = df["messages"].apply(lambda row: row[0]["content"])
    pattern1 = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    df["Reasoning"] = df["reannotated_assistant_content"].apply(
        lambda x: pattern1.search(x).group(1) if pattern1.search(x) else None
    )

    pattern2 = re.compile(r"</think>(.*)", re.DOTALL)
    df["Assistant"] = df["reannotated_assistant_content"].apply(
        lambda x: pattern2.search(x).group(1).strip() if pattern2.search(x) else None
    )
    df.drop(
        columns=[
            "id",
            "source",
            "verified",
            "quality_metrics",
            "reannotated_assistant_content",
            "reannotated_messages",
            "source_dataset",
            "messages",
        ],
        inplace=True,
    )
    order_reasoning_df(df)
    df["Language"] = language_to_string(Language.ENG)
    return df


def format_big_russian(df):
    df[
        "System"
    ] = """Ты виртуальный ассистент. Ты отвечаешь на вопросы людей, помогаешь им и поддерживаешь. Ты создан, чтобы быть полезным, безобидным и честным. Ты отвечаешь на том языке, на котором был задан вопрос или попросил пользователь.

Answer in the following format:
<think>Reasoning: ...</think>
..."""
    df.rename(columns={"question": "User"}, inplace=True)
    pattern1 = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    df["Reasoning"] = df["conversation"].apply(
        lambda x: pattern1.search(x[2]["content"]).group(1)
        if pattern1.search(x[2]["content"])
        else None
    )

    pattern2 = re.compile(r"</think>(.*)", re.DOTALL)
    df["Assistant"] = df["conversation"].apply(
        lambda x: pattern2.search(x[2]["content"]).group(1).strip()
        if pattern2.search(x[2]["content"])
        else None
    )
    order_reasoning_df(df)
    df["Language"] = language_to_string(Language.RU)
    df = df[["System", "User", "Reasoning", "Assistant", "Language"]]
    return df

def save_dataframe(df, filename, need_header=False):
    df.to_csv(filename, mode="a", header=need_header, index=False)
    del df
    gc.collect()

def save_dataset(
    ds,
    ds_type,
    filename,
    format_ds,
    *args,
    batch_size=100000,
    need_header=True,
    max_file_size=-1,
    index=0,
):
    df = create_df(ds_type)
    current_index = index
    save_dataframe(df, filename, need_header=need_header)
    del df
    for i, batch in enumerate(ds["train"].to_pandas(batch_size=batch_size, batched=True)):
        df_batch = format_ds(batch, *args)
        n = len(df_batch)

        df_batch.index = range(current_index, current_index + n)
        current_index += n
        save_dataframe(df_batch, filename)
        del df_batch, batch
        gc.collect()
        if max_file_size != -1:
            file_size = os.path.getsize(filename)
            if file_size > max_file_size:
                return current_index
    return current_index
