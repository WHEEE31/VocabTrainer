# this is for all the processing steps to go from the raw dataset to the finished model + polished dataset
import os
import re
import csv
import ast
import json
import math
from sklearn.neighbors import NearestNeighbors
import tqdm
import numpy
import random
import pandas
import sqlite3
import fasttext
import wordfreq
from sklearn.metrics.pairwise import cosine_similarity
#import textstat
#import pronouncing

ROOT = "C:\\Users\\super\\Documents\\projects\\VOCABULATOR"
FEATURES_PATH = os.path.join(ROOT, "assets", "database", "WORDSfeatures.csv")
JSONL_PATH = os.path.join(ROOT, "assets", "database", "WORDS.jsonl")
EMBED_PATH = os.path.join(ROOT, "assets", "database", "embeddings", "cc.en.300.bin")
FREQ_PATH = os.path.join(ROOT, "assets", "database", "frequency")

class LabelingTools():
    """
    csv title line (line 1):
    word,rtg
    """
    PATH = os.path.join(ROOT, "assets", "database", "WORDSlabeling.csv")

    @classmethod
    def init(cls):
        with open(cls.PATH, "r", newline='', encoding='utf-8') as f:
            header = csv.DictReader(f).fieldnames
        
        rows = []
        with open(os.path.join(ROOT, "assets", "database", "WORDSprogress.csv"), "r", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['defined']=='True': rows.append({'word':row['word'], 'rtg':''})
        with open(cls.PATH, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
    
    @classmethod
    def base(cls, func):
        updated = []; header = []
        with open(cls.PATH, "r", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            for line in reader:
                out = func(line)
                updated.append(line if out is None else out)

        with open(cls.PATH, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(updated)   

    @classmethod
    def add(cls, num, words = None):
        """
        LabelingTools.add(num, words = None) --> None
            adds num to each rating present (does nothing if no words have ratings yet)
            words is an optional list param; if filled, only the specified words will be affected
        """
        def temp(line):
            if line['rtg'] == '': return
            if words is not None and line['word'] not in words: return
            return {'word':line['word'], 'rtg':str(float(line['rtg']) + num)}

        cls.base(temp)

    @classmethod
    def multiply(cls, num, words = None):
        """
        LabelingTools.multiply(num, words = None) --> None
            multiplies each rating present by num (does nothing if no words have ratings yet)
            words is an optional list param; if filled, only the specified words will be affected
        """
        def temp(line):
            if line['rtg'] == '': return
            if words is not None and line['word'] not in words: return
            return {'word':line['word'], 'rtg':str(float(line['rtg']) * num)}

        cls.base(temp)

    @classmethod
    def reset(cls, num = None):
        """
        LabelingTools.reset(num = None) --> None
            deletes all existing ratings
            if num is specified, sets all words' ratings to num instead
        """
        def temp(line):
            return {'word':line['word'], 'rtg':num if num else ''}

        cls.base(temp)

    @classmethod
    def sample(cls):
        def flatten(tree):
            out = []
            def dfs(node):
                if isinstance(node, list):
                    # It's a "leaf list" if none of its elements are lists
                    if all(not isinstance(e, list) for e in node):
                        out.append(node)
                    else:
                        for e in node:
                            dfs(e)
            dfs(tree)
            return out
        
        print('Executing binning - frequency and # syllables')
        # syl: (1-2), (3-4), (5-6), (7-10)                  freq: log scale (1-10), (100-1000), ... (10^8-10^9)
        categories = [[[] for syl_bin in range(4)] for freq_bin in range(9)]
        with open(FEATURES_PATH, "r", encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in tqdm.tqdm(reader, total=161256):
                w = line['word']
                f = int(line['frequency'])
                s = int(line['# syllables'])
                categories[int(math.log10(f))][(s-1)//2 if s<9 else 3].append(w)
        
        print('Executing stratified sampling')
        bins = flatten(categories)
        rng = random.Random(21); sampled = []
        min_per_bin = 5; max_per_bin = 100; subset_percent = 0.8
        for b in tqdm.tqdm(bins):
            n = min(
                max(min_per_bin, math.ceil(subset_percent * len(b))),
                max_per_bin,
                len(b)
            )
            if n > 0:
                sampled.extend(rng.sample(b, n))

        print('Saving sampled words to WORDSlabeling.csv')
        out = [{'word':w, 'rtg':''} for w in sampled]
        with open(cls.PATH, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=out[0].keys())
            writer.writeheader()
            writer.writerows(out)

def EXTRACT_AOA():
    in_path = os.path.join(ROOT, "assets", "database", "truth", "aoa30k.xlsx")
    out_path = os.path.join(ROOT, "assets", "database", "WORDSlabeling.csv")

    df = pandas.read_excel(in_path, sheet_name=0)

    words_data = []
    for _, row in tqdm.tqdm(df.iterrows(),total=31124):
        word = str(row['Word']).strip()
        rating = row['Rating.Mean']

        if pandas.isna(rating):
            continue
        if isinstance(rating, str) and rating.strip().upper() == "#N/A":
            continue

        words_data.append({'word': word, 'rtg': f'{rating:.2f}'})

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['word', 'rtg'])
        writer.writeheader()
        writer.writerows(words_data)

    print("Extracted", len(words_data), "words with AoA ratings.")

class FeatureExtract():
    def __init__(self):
        numpy.set_printoptions(edgeitems=1,linewidth=1000)
        
        print('Initializing WordFrequency array')
        amounts = {'en_full_2016.txt' : 534751778,
                    'en_full_2018.txt' : 734777659}
        self.wordfreq = {}
        for text in ['en_full_2016.txt', 'en_full_2018.txt']:
            with open(os.path.join(FREQ_PATH, text), "r", encoding='utf-8') as f:
                for l in tqdm.tqdm(f, total=amounts[text]):
                    parts = l.strip().split()
                    self.wordfreq[parts[0]] = int(int(parts[1])/amounts[text] * 10**9)
        print('Initializing Fasttext embedding model')
        self.ebd_model = fasttext.load_model(EMBED_PATH)
    
    def _run(self, lim=1):
        print('Extracting features')
        out = []
        with open(JSONL_PATH, "r", newline='', encoding='utf-8') as f:
            for line in tqdm.tqdm(f, total=161256):
                if random.random() > lim: continue
                entry = json.loads(line)
                word = next(iter(entry))
                data = entry[word]
                out.append(self.extract_features(word, data))

        print('Saving data to WORDSfeatures.csv')
        with open(FEATURES_PATH, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=out[0].keys())
            writer.writeheader()
            writer.writerows(out)
        print('Feature extraction completed successfully')

    def extract_features(self, word, data):
        """
        EXPLICITLY DERIVED
        - number of characters
        - number of syllables
        - vowel:consonant ratio
        - ?? count of morphemes (prefixes, roots, suffixes)
        IMPLICITLY DERIVED
        - number of definitions
        - number of parts of speech
        LOOKUP
        - embedding
        - frequency rank
        - ?? frequency rank in academic contexts
        """
        n_char = self.get_nchar(word)

        n_syl = self.get_nsyl(word)

        vcr = self.get_vcr(word)

        n_def, n_pos = self.get_ndefpos(data)

        freq = self.get_freq(word)

        ebd = self.get_ebd(word)

        return {'word':word, '# characters':str(n_char), '# syllables':str(n_syl), 'vowel:consonant':str(vcr), 
                '# definitions':str(n_def), '# parts of speech':str(n_pos), 'frequency':str(freq), 'embedding':ebd}
    
    def get_nchar(self, word):
        return len(word)
    
    def get_nsyl(self, word):
        word = word.lower().strip()
        original_word = word
        word = re.sub(r'[^a-z]', '', word)                                              # Remove non-alpha characters
        if not word:
            return 0
        exception_add = {                                                               # Known exceptions where rules don't apply
            'serious', 'crucial', 'poem', 'family', 'business', 'quiet', 'science', 'vehicle',
            'doing', 'being', 'seeing', 'area', 'idea', 'create', 'real', 'riot', 'giant',
            'diary', 'piano', 'theatre'
        }
        exception_del = {
            'every', 'interest', 'chocolate', 'different', 'camera', 'general',
            'restaurant', 'comfortable', 'temperature', 'separate'
        }

        syllables = len(re.findall(r'[aeiouy]+', word))
        if word.endswith("e") and not word.endswith(("le", "ue", "ye", "oe")):          # Subtract silent 'e'
            syllables -= 1
        if re.search(r'[^aeiou]le$', word):                                             # Add for "consonant + le" endings (e.g., "table", "cable")
            syllables += 1
        if word.startswith("mc"):                                                       # Fix for "mc" prefixes (e.g., "McDonald")
            syllables += 1
        diphthongs = ["ia", "io", "ea", "eo", "eu", "ie", "ai", "oi", "ou", "ui"]       # Diphthongs (common vowel pairs often forming one syllable)
        for pair in diphthongs:
            if pair in word:
                syllables -= 1
        for pair in ['ia', 'io', 'eo']:                                                 # Re-correct certain split-vowel cases (e.g., "naïve", "theatre")
            if pair in word:
                syllables += 1                                            
        if original_word in exception_add:                                              # Apply manual exception list overrides
            syllables += 1
        if original_word in exception_del:
            syllables -= 1

        return max(1, syllables)
    
    def get_vcr(self, word):
        _temp_vowelcount = sum(c in set(['a','e','i','o','u']) for c in word)
        _temp_consonantcount = sum(c in set(['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z']) for c in word)
        vcr = _temp_vowelcount / _temp_consonantcount if _temp_consonantcount != 0 else 0.61    # 0.61 is the average v:c ratio across all words
        return f'{vcr:.5f}'

    def get_ndefpos(self, data):
        d=0; p=0
        for _pos in data:
            p += 1
            d += len(_pos['definitions'])
        return d, p
    
    def get_freq(self, word):
        out = int(wordfreq.word_frequency(word, 'en', 'large') * (10**9))
        if out == 0:
            out = self.wordfreq.get(word)
        return math.log(out) if out else 0
    
    def get_ebd(self, word):
        raw = self.ebd_model.get_word_vector(word)
        string = ",".join([f"{x}" for x in raw])
        return string
    
    def embedding_to_numpy(self, embedding):
        parts = embedding.strip().split()
        word = parts[0] 
        vec = numpy.array(parts[1:], dtype='float32')
        return word, vec

class FinalDataset:

    def __init__(self):
        self.conn = sqlite3.connect(os.path.join(ROOT, "assets", "database", "database.db"))
        self.cursor = self.conn.cursor()

        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS words (
                                word TEXT PRIMARY KEY,
                                difficulty REAL,
                                definitions TEXT,
                                synonyms TEXT
                            )
                            """)
        self.cursor.execute("DELETE FROM words")

        '''        
        word: {difficulty: int, definitions : [[pos, def, ex]], synonyms: [string, string, string]}        
        '''

    def execute(self, TEST=False):
        print('creating final dataset')
        dif_thresh = 1
        synonym_thresh = 3

        def protect_NaNesque_words(df):
            df['word'] = df['word'].apply(lambda x: f'__LITERAL_{x}' if isinstance(x, str) and x in ['nan', 'Nan', 'none', 'null'] else x)
        
        feature_df = pandas.read_csv(FEATURES_PATH, keep_default_na=False, na_values=[])
        difficulty_df = pandas.read_csv(os.path.join(ROOT, "assets", "database", "WORDSdifficulties.csv"), keep_default_na=False, na_values=[])
        protect_NaNesque_words(feature_df)
        protect_NaNesque_words(difficulty_df)

        def temp(string):
            return numpy.array(ast.literal_eval(string), dtype=float)
        feature_df['embedding'] = feature_df['embedding'].apply(temp)
        feature_df = feature_df.dropna(subset=['embedding'])
        merged_df = pandas.merge(
            feature_df[['word', 'embedding']],
            difficulty_df[['word', 'difficulty']],
            on='word',
            how='inner'
        )
        embeds = merged_df[['word', 'embedding', 'difficulty']]
        embeds.set_index("word", inplace=True)
        embeds.index = embeds.index.map(lambda x: x[10:] if x[:10] == '__LITERAL_' else x)

        X_all = numpy.stack(embeds['embedding'].values)
        word_list = embeds.index.to_list()
        nn_model = NearestNeighbors(n_neighbors=30, metric='cosine')
        nn_model.fit(X_all)

        out = []; test_percent=0.01
        with open(JSONL_PATH, "r", encoding='utf-8') as f:
            for i, line in tqdm.tqdm(enumerate(f), total=161257):
                if (random.random() >= test_percent) and TEST: continue
                entry = json.loads(line)
                word = next(iter(entry))
                data = entry[word]

                try:
                    dif = embeds.loc[word]['difficulty']
                except Exception:
                    continue
                if dif < dif_thresh: continue

                seen_pos = set()
                defs = []
                for entry in data:
                    pos = entry.get('partOfSpeech')
                    if pos in seen_pos:
                        continue
                    for definition_entry in entry.get('definitions', []):
                        definition = definition_entry.get('definition')
                        examples = definition_entry.get('examples', [])
                        example = examples[0] if examples else None
                        defs.append([pos, definition, example])
                        seen_pos.add(pos)
                        break
                    if len(defs) == 2:
                        break
                
                emb = embeds.loc[word]['embedding'].reshape(1, -1)
                distances, indices = nn_model.kneighbors(emb, n_neighbors=30)
                syns = []
                for idx, dist in zip(indices[0], distances[0]):
                    similar_word = word_list[idx]
                    if similar_word == word:
                        continue
                    if embeds.loc[similar_word]['difficulty'] > synonym_thresh:
                        continue
                    sim_score = 1 - dist  # cosine similarity = 1 - distance
                    syns.append((similar_word, sim_score) if TEST else similar_word)
                    if len(syns) == 3:
                        break

                out.append({'word' : word, 'difficulty': dif, 'definitions' : defs, 'synonyms' : syns})

        # for debugging
        with open(os.path.join(ROOT, "assets", "database", "DATABASE.csv"), mode = "w", newline = '', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames = out[0].keys())
            writer.writeheader()
            writer.writerows(out)

        for line in tqdm.tqdm(out, desc='writing to sql database'):
            self.cursor.execute("""
                INSERT OR REPLACE INTO words (word, difficulty, definitions, synonyms)
                VALUES (?, ?, ?, ?)
            """, (
                line['word'],
                line['difficulty'],
                json.dumps(line['definitions']),
                json.dumps(line['synonyms'])
            ))

        self.conn.commit()
        self.conn.close()

    '''
    Example loading from SQL database:
        conn = sqlite3.connect("word_database.db")
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM words WHERE word = ?", ("example",))
        row = cursor.fetchone()
        if row:
            word, embedding_json, pos, definition, frequency, synonyms_json = row
            embedding = json.loads(embedding_json)
            synonyms = json.loads(synonyms_json)
    '''
        
def find_cool_stuff():
    max_syl = 0
    max_word = ''
    with open(FEATURES_PATH, "r", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        #print(max(l['number-of-definitions'] for l in reader))
        for line in reader:
            if int(line['# syllables']) >= max_syl:
                max_syl = int(line['# syllables'])
                max_word = line['word']
    print(f'{max_word}--> {max_syl}')
    return

def find_percent(x):
    with open(os.path.join(ROOT, "assets", "database", "WORDSdifficulties.csv"), "r", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        good = 0; total = 0
        for line in reader:
            total += 1
            if int(line['difficulty']) <= x:
                good += 1

        print(f"{good/total*100:.2f}% [{good}/{total}]")

def extract():
    f = FeatureExtract()
    f._run()

def sample():
    LabelingTools.sample()

def get_final_dataset():
    fd = FinalDataset()
    fd.execute()

def test(word):
    w = word.lower()
    w = re.sub(r'[^a-z]', '', w)
    matches = re.findall(r'[aeiouy]+', w)
    return max(1, len(matches))

def test2():
    with open(FEATURES_PATH, "r", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        counts = {'total':0,
                  'no embedding':0,
                  'no frequency':0}
        for line in reader:
            counts['total'] += 1
            if line['embedding'] == '':
                counts['no embedding'] += 1
            if line['frequency'] == '':
                counts['no frequency'] += 1
        print(f'Out of {counts['total']} words, \n{counts['no frequency']} are missing a frequency metric, and\n{counts['no embedding']} are missing an embedding.')

def test3():
    f = FeatureExtract()
    while True:
        x = input()
        print(f.get_freq(x))

def lookup(target):
    definitions = {}
    with open(JSONL_PATH, "r", encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            word = next(iter(entry))
            if word == target:
                return entry[word]
    return None

def looktwo():
    while True:
        t = input("Word to lookup...")
        dat = lookup(t)
        if dat is None:
            print('    Invalid word, skipping...')
            continue
        print(f'    Part of speech: {dat[0]['partOfSpeech']}\n    Definition: {dat[0]['definitions'][0]['definition']}\n    Example: {dat[0]['definitions'][0]['examples'][0] if len(dat[0]['definitions'][0]['examples']) > 0 else 'none'}')

get_final_dataset()