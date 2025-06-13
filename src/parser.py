from datasets import load_dataset
import requests
import json
import re
import pandas as pd
import os

os.makedirs("./tests", exist_ok=True)

ds = (
    load_dataset("MatrixStudio/Codeforces-Python-Submissions")
    .filter(lambda row: row["contestId"] > 400)
    .sort("contestId")
)
contest_df = pd.DataFrame(
    columns=["Contest", "Name", "Description", "TestFile", "Score"]
)
tests_df = pd.DataFrame(columns=["Name", "Tests"])
ratings_df = pd.DataFrame(
    columns=["Contest", "Username", "Score", "Rating", "RatingChange"]
)

pattern = r"^(?!.*Div\. 1)(?!.*Div\.[0-9].*Div\.[0-9])(.*(Div\. [23]).*)$"

handle_score = {}
current_contest_id = -1
contest_name = ""

names = set()
MAX_DF_SIZE = 200

for df in ds["train"].to_pandas(batch_size=100, batched=True):
    if len(contest_df) >= MAX_DF_SIZE:
        break
    for i in range(len(df)):
        if len(contest_df) >= MAX_DF_SIZE:
            break
        contest_id = df.iloc[i]["contestId"]
        problem_name = df.iloc[i]["name"]
        if problem_name in names:
            continue
        if contest_id == current_contest_id:
            if re.search(pattern, contest_name) is None:
                continue
        else:
            names.clear()
            r = requests.get(
                f"https://codeforces.com/api/contest.standings?contestId={contest_id}"
            )
            if r.status_code != 200:
                continue
            current_contest_id = contest_id
            contest_name = r.json()["result"]["contest"]["name"]
            if re.search(pattern, contest_name) is None:
                continue
            ranklist_rows = r.json()["result"]["rows"]
        for ranklist in ranklist_rows:
            members = ranklist["party"]["members"]
            if len(members) > 1:
                continue
            handle = members[0]["handle"]
            handle_score[handle] = ranklist["points"]
        r = requests.get(
            f"https://codeforces.com/api/contest.ratingChanges?contestId={contest_id}"
        )
        if r.status_code != 200:
            continue
        rating_changes = r.json()["result"]
        for rating_change in rating_changes:
            handle = rating_change["handle"]
            if handle_score.get(handle) is None:
                continue
            new_row3 = pd.Series(
                [
                    contest_name,
                    handle,
                    handle_score[handle],
                    rating_change["newRating"],
                    rating_change["newRating"] - rating_change["oldRating"],
                ],
                index=ratings_df.columns,
            )
            ratings_df.loc[ratings_df.size] = new_row3
        tests = df.iloc[i]["test_cases"]
        names.add(problem_name)
        filename = (
            "."
            + f"/tests/{contest_name}_{problem_name}".replace(" ", "_")
            .replace(",", "")
            .replace(".", "")
            + ".json"
        )

        with open(filename, "w") as output:
            output.write(json.dumps(tests.tolist()))
        new_row1 = pd.Series(
            [
                contest_name,
                problem_name,
                df.iloc[i]["problem-description"],
                filename,
                df.iloc[i]["points"],
            ],
            index=contest_df.columns,
        )
        new_row2 = pd.Series([problem_name, tests], index=tests_df.columns)
        contest_df.loc[contest_df.size] = new_row1
        tests_df.loc[tests_df.size] = new_row2

contest_df = contest_df.reset_index().drop(columns=["index"])
tests_df = tests_df.reset_index().drop(columns=["index"])
ratings_df = ratings_df.reset_index().drop(columns=["index"])

contest_df.to_csv("contest_df.csv")
tests_df.to_csv("tests_df.csv")
ratings_df.to_csv("ratings_df.csv")
