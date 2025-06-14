from typing import List
import math

def calculate_seed(contest_participants_ratings: List[int], current_rating: int, participant):
    seed = 1
    for i in range(len(contest_participants_ratings)):
        if participant and current_rating == contest_participants_ratings[i]:
            participant = False
        else:
            seed += 1 / (1 + 10 ** ((current_rating - contest_participants_ratings[i]) / 400))
    return seed

def calculate_raw_d(contest_participants_ratings: List[int], current_rating: int, place: int, participant):
    seed = calculate_seed(contest_participants_ratings, current_rating, participant)
    mean = math.sqrt(place * seed)
    expected_rating = 0
    l = 0
    r = 20000
    while abs(seed - mean) > 0.0001:
        expected_rating = (r + l) / 2
        seed = calculate_seed(contest_participants_ratings, expected_rating, participant)
        if seed < mean:
            r = expected_rating
            continue
        if seed > mean:
            l = expected_rating
            continue
        if seed == mean:
            break
    d = (expected_rating - current_rating) / 2
    return d

def calculate_rating_change(contest_participants_ratings, current_rating: int, place: int, d_sum, s_sum, n, s, participant):
    d = calculate_raw_d(contest_participants_ratings, current_rating, place, participant)
    rating_change = d

    #increments
    inc1 = (-d_sum/n) - 1
    rating_change += inc1
    inc2 = min(max(-s_sum/s, -10), 0)
    rating_change += inc2

    return int(rating_change)

def parse_ratings(contest_participants_ratings_places: List[tuple[int,int]], current_rating: int, place: int, participant):
    n = len(contest_participants_ratings_places)
    contest_participants_ratings_places = list(reversed(sorted(contest_participants_ratings_places)))
    contest_participants_ratings = []
    for tuple in contest_participants_ratings_places:
        contest_participants_ratings.append(tuple[0])
    s = min(n, int(4 * math.sqrt(n)))
    s_sum = 0
    d_sum = 0
    for i in range(n):
        d_i = calculate_raw_d(contest_participants_ratings, contest_participants_ratings_places[i][0], contest_participants_ratings_places[i][1], participant)
        d_sum += d_i
        if i < s:
            s_sum += d_i
    changes = []
    for target in contest_participants_ratings_places:
        rating_change = calculate_rating_change(contest_participants_ratings, target[0], target[1], d_sum, s_sum, n, s, True)
        changes.append([target[0], target[1], rating_change])
    rating_change = calculate_rating_change(contest_participants_ratings, current_rating, place, d_sum, s_sum, n, s, participant)

    #final assertions
    for j in range(len(changes)):
        rating_a = current_rating
        rating_b = changes[j][0]
        place_a = place
        place_b = changes[j][1]
        change_a = rating_change
        change_b = changes[j][2]
        if rating_a < rating_b and place_a < place_b:
            if rating_a + change_a > rating_b + change_b:
                rating_change -= (rating_a + change_a) - (rating_b + change_b)
        if rating_a < rating_b and place_a > place_b:
            if change_a < change_b:
                rating_change = change_b

    #performance calculation
    '''
    l = 0
    r = 20000
    performance = 10000
    expected_change = 1
    while expected_change != 0:
        performance = (l + r) / 2
        expected_change = calculate_rating_change(contest_participants_ratings, performance, place, d_sum, s_sum, n, s)
        if expected_change < 0:
            r = performance
        elif expected_change > 0:
            l = performance
        else:
            break
    '''
    return rating_change#, performance

"""
import pandas as pd
import time
data = pd.read_csv('DF3.csv')

for column in data.columns:
    globals()[column] = data[column].values

rp = []
count = 0
sumcount = 0
for i in range(len(Rating)):
    if Contest[i] == "Codeforces Round 334 (Div. 2)":
        rp.append((int(Rating[i]), int(place[i]) + 1))
start = time.time()
test = rp.copy()
changes = parse_ratings(test, test[0][0], test[0][1], True)
print(changes)
print(RatingChange[0])
"""