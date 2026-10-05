# as the learner isn't working right now, this is a temporary solution
# it will go through the WORDSdifficulties and the WORDS, and get the word-definition and word-synonym pairs
# for each word that lies within a difficulty range.
# it will put these into a text file for me to upload to knowt

import datetime
import sqlite3
import random
import pickle
import pandas
import tqdm
import json
import csv
import os

ROOT = "C:\\Users\\super\\Documents\\projects\\VOCABULATOR"

conn = sqlite3.connect(os.path.join(ROOT, "assets", "database", "database.db"))
cursor = conn.cursor()

THRESH = [15, 20]
test = 1

csv_out = []
txt_out = []
just_words = []
delimiter = '~~'

percent = [0,0]

with open(os.path.join(ROOT, "src", "learner", "debugging", "avoid.txt"), mode='r', encoding='utf-8', newline='\n') as f:
    bad = f.read().splitlines()

with open(os.path.join(ROOT, "assets", "database", "WORDSdifficulties.csv"), mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in tqdm.tqdm(reader):
        percent[0] += 1
        word = row['word']
        diff = eval(row['difficulty'])

        if diff < THRESH[0] or diff > THRESH[1]:
            continue
        if random.random() >= test:
            continue
        if word in bad:
            continue

        cursor.execute("SELECT * FROM words WHERE word = ?", (word,))
        r = cursor.fetchone()
        if r is None:
            continue
        _, _, defs, syns = r
        temp = []
        for d in json.loads(defs):
            if d[0] != 'noun':
                temp.append(f'({d[0]}) {d[1]}')
        if len(temp) == 0:
            continue
        defs = temp
        syns = json.loads(syns)

        just_words.append(word)

        for i in defs:
            csv_out.append({'Q' : word, 'A' : i})
            txt_out.append(word + delimiter + i)
        percent[1] += 1

print(f'{percent[1] / percent[0]*100:.2f}% [{percent[1]}/{percent[0]}]')

with open(os.path.join(ROOT, "src", "learner", "debugging", f"{datetime.date.today()}_debug_out.csv"), mode='w', encoding='utf-8', newline='\n') as f:
    writer = csv.DictWriter(f, fieldnames=csv_out[0].keys())
    writer.writeheader()
    writer.writerows(csv_out)

with open(os.path.join(ROOT, "src", "learner", "debugging", "knowt_out.txt"), mode='w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(txt_out))

with open(os.path.join(ROOT, "src", "learner", "debugging", "just_words.txt"), mode='w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(just_words))