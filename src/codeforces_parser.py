from datasets import load_dataset
from utils.parsers_utils import save_dataframe
import requests
import json
import re
import pandas as pd
import os
import psutil

os.makedirs("./tests", exist_ok=True)

while True:
    user_input = input("Введите количество контестов [не больше 200]: ")
    if not user_input:
        continue
    try:
        CONTEST_COUNT = int(user_input)
        break
    except:
        print('Неверный формат, требуется "%d"')
        continue

ds = (
    load_dataset("MatrixStudio/Codeforces-Python-Submissions")
    .sort(column_names=["contestId", "points"])
    .filter(lambda row: row["contestId"] > 400)
)
contest_df = pd.DataFrame(
    columns=[
        "Contest",
        "Name",
        "Description",
        "TestFile",
        "Score",
        "Solution",
        "Status",
    ]
)
tests_df = pd.DataFrame(columns=["Name", "Tests"])
ratings_df = pd.DataFrame(
    columns=["Contest", "Username", "Score", "Rating", "RatingChange"]
)

pattern = r"^(?!.*Div\. 1)(?!.*Div\.[0-9].*Div\.[0-9])(.*(Div\. [23]).*)$"

handle_score = {}
current_contest_id = -1
contests = set()
current_problem = ""
is_contest_valid = False
contest_df.to_csv("contest_df.csv")
tests_df.to_csv("tests_df.csv")
ratings_df.to_csv("ratings_df.csv")
for df in ds["train"].to_pandas(batch_size=100000, batched=True):
    if psutil.virtual_memory().percent > 85.0:
        save_dataframe(contest_df, "contest_df.csv")
        save_dataframe(tests_df, "tests_df.csv")
        save_dataframe(ratings_df, "ratings_df.csv")
        contest_df = pd.DataFrame(
            columns=[
                "Contest",
                "Name",
                "Description",
                "TestFile",
                "Score",
                "Solution",
                "Status",
            ]
        )
        tests_df = pd.DataFrame(columns=["Name", "Tests"])
        ratings_df = pd.DataFrame(
            columns=["Contest", "Username", "Score", "Rating", "RatingChange"]
        )
    if len(contests) > CONTEST_COUNT:
        break
    for i in range(len(df)):
        if len(contests) > CONTEST_COUNT:
            break
        contest_id = df.iloc[i]["contestId"]
        problem_name = df.iloc[i]["name"]
        if contest_id == current_contest_id:
            if not is_contest_valid:
                continue
        else:
            r = requests.get(
                f"https://codeforces.com/api/contest.standings?contestId={contest_id}"
            )
            if r.status_code != 200:
                continue
            current_contest_id = contest_id
            contest_name = r.json()["result"]["contest"]["name"]
            is_contest_valid = not (re.search(pattern, contest_name) is None)
            if not is_contest_valid:
                continue
            ranklist_rows = r.json()["result"]["rows"]
        for ranklist in ranklist_rows:
            members = ranklist["party"]["members"]
            if len(members) > 1:
                continue
            handle = members[0]["handle"]
            handle_score[handle] = ranklist["points"]
        if contest_id not in contests:
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
            contests.add(contest_id)
        tests = df.iloc[i]["test_cases"]
        if problem_name != current_problem:
            current_problem = problem_name
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
                current_problem,
                df.iloc[i]["problem-description"],
                filename,
                df.iloc[i]["points"],
                df.iloc[i]["code"],
                df.iloc[i]["verdict"],
            ],
            index=contest_df.columns,
        )
        new_row2 = pd.Series([problem_name, tests], index=tests_df.columns)
        contest_df.loc[contest_df.size] = new_row1
        if len(tests_df) == 0:
            tests_df.loc[len(tests_df)] = new_row2
        elif tests_df.loc[len(tests_df) - 1]["Name"] != problem_name:
            tests_df.loc[len(tests_df)] = new_row2

contest_df = contest_df.reset_index().drop(columns=["index"])
tests_df = tests_df.reset_index().drop(columns=["index"])
ratings_df = ratings_df.reset_index().drop(columns=["index"])

contest_df.to_csv("contest_df.csv", header=False)
tests_df.to_csv("tests_df.csv", header=False)
ratings_df.to_csv("ratings_df.csv")
