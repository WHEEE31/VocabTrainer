'''

dataset.py

Makes the dataset that is used in the rest of the app. 
Defines each word with Wiktionary.com entries.
Word sources:
 - nltk (nltk.corpus --> words.words())

'''

import csv
import os
import re
import sys
import html
import json
import nltk
import time
import random
import requests
import traceback
import unicodedata
from tqdm import tqdm
import wikitextparser as wtp
from nltk.corpus import words
from wordfreq import iter_wordlist
from difflib import SequenceMatcher

# Parts Of Speech to be included in the generated words list
POS = {
    # lexical - common & specialized
    "adjective"              : True,
    "adverb"                 : True,
    "ambiposition"           : False,
    "article"                : False,
    "circumposition"         : False,
    "classifier"             : False,
    "conjunction"            : False,
    "contraction"            : False,
    "counter"                : False,
    "determiner"             : False,
    "ideophone"              : False,
    "interjection"           : True,
    "noun"                   : True,
    "numeral"                : True,
    "participle"             : False,
    "particle"               : False,
    "postposition"           : False,
    "preposition"            : False,
    "pronoun"                : False,
    "proper noun"            : True,
    "verb"                   : True,
    # morpheme-level headers
    "circumfix"              : False,
    "combining form"         : False,
    "inflex"                 : False,
    "interfix"               : False,
    "prefix"                 : False,
    "root"                   : False,
    "suffix"                 : False,
    # symbols, characters & related
    "diacritical mark"       : False,
    "letter"                 : False,
    "ligature"               : False,
    "number"                 : False,
    "punctuation mark"       : False,
    "syllable"               : False,
    "symbol"                 : False,
    # phrases & multi-word units
    "phrase"                 : True,
    "prepositional phrase"   : False,
    "proverb"                : False,
    # language-specific / logographic
    "han character"          : False,
    "romanization"           : False,
    "logogram"               : False,
    "determinative"          : False
}; valid_pos = {i for i in POS if POS[i]}

_unicode_map = {
    '\u00e6': 'ae',    # æ
    '\u0153': 'oe',    # œ
    '\u00df': 'ss',    # ß
    '\u017f': 's',     # ſ (long s)
    # Accented characters → ASCII equivalents
    '\u00e0': 'a',     # à
    '\u00e1': 'a',     # á
    '\u00e2': 'a',     # â
    '\u00e3': 'a',     # ã
    '\u00e4': 'a',     # ä
    '\u00e5': 'a',     # å
    '\u00e7': 'c',     # ç
    '\u00e8': 'e',     # è
    '\u00e9': 'e',     # é
    '\u00ea': 'e',     # ê
    '\u00eb': 'e',     # ë
    '\u00ec': 'i',     # ì
    '\u00ed': 'i',     # í
    '\u00ee': 'i',     # î
    '\u00ef': 'i',     # ï
    '\u00f1': 'n',     # ñ
    '\u00f2': 'o',     # ò
    '\u00f3': 'o',     # ó
    '\u00f4': 'o',     # ô
    '\u00f5': 'o',     # õ
    '\u00f6': 'o',     # ö
    '\u00f9': 'u',     # ù
    '\u00fa': 'u',     # ú
    '\u00fb': 'u',     # û
    '\u00fc': 'u',     # ü
    '\u00fd': 'y',     # ý
    '\u00ff': 'y',     # ÿ
    # Smart punctuation → ASCII
    '\u2018': "'",     # ‘
    '\u2019': "'",     # ’
    '\u201c': '"',     # “
    '\u201d': '"',     # ”
    '\u2013': '-',     # –
    '\u2014': '-',     # —
    '\u2026': '...',   # …
    '\u2032': "'",
    # Misc symbols
    '\u00a0': ' ',     # Non-breaking space
    '\u00b7': '*',     # Middle dot (can be kept or dropped)
    '\u2122': '(TM)',  # Trademark ™
    '\u00ae': '(R)',   # ®
    '\u00a9': '(C)',   # ©
    '\u00bc': '1/4',   # ¼
    '\u00bd': '1/2',   # ½
    '\u00be': '3/4',   # ¾
    # Latin extended ligatures and symbols
    '\u0131': 'i',     # ı (dotless i)
    '\u0142': 'l',     # ł
    '\u00fe': 'th',    # þ
    '\u00f0': 'd',     # ð
    '\u0107': 'c',     # ć
    '\u0111': 'd',     # đ
    '\u015f': 's',     # ş
}

def clean_wikitext(text, word, is_example):
    def read_balanced(s: str, start, open_char, close_char):
        depth = 1
        i = start
        buf = []
        if close_char == "'''": s = s.replace("'''", "@@@"); close_char = "@@@"
        while i < len(s):
            if s[i:i+len(open_char)] == open_char:
                depth += 1
                buf.append(open_char)
                i += len(open_char)
            elif s[i:i+len(close_char)] == close_char:
                depth -= 1
                if depth == 0:
                    return ''.join(buf), i + len(close_char)
                buf.append(close_char)
                i += len(close_char)
            else:
                buf.append(s[i])
                i += 1
        return ''.join(buf), i

    def process_template(raw):
        parts = raw.split('|')
        name = parts[0].strip().lower()
        if name in ('w', 'wikipedia'):
            return parts[1] if len(parts) > 1 else parts[0]
        if name == 'u':
            return parts[1] if len(parts) > 1 else ''
        if name in ('taxfmt', 'taxlink'):
            return parts[1] if len(parts) > 1 else ''
        if name == 'non-gloss':
            return parts[1] if len(parts) > 1 else ''
        if name == '...':
            return '...'
        if name == 'lb':
            return ''
        if name == 'sic':
            return '[sic]'
        return ''

    def process_link(raw):
        if '|' in raw:
            return raw.split('|')[-1]
        if raw in ['...', 'sic']:
            return raw
        return raw
    
    def process_html(raw):
        if raw == 'br':
            return '\n'
        return ''
        
    i = 0
    found_word = False
    out = []
    while i < len(text):
        if text[i:i+2] == '{{':                                     # templates
            inner, i = read_balanced(text, i+2, '{{', '}}')
            out.append(process_template(inner))
        elif text[i:i+2] == '[[':                                   # links
            inner, i = read_balanced(text, i+2, '[[', ']]')
            out.append(process_link(inner))
        elif text[i] == '[':                                        # weird links
            if i<len(text)-4 and text[i+1:i+4] == '...':
                out.append('...'); i += 5
            else:
                out.append(text[i]); i += 1
        elif text[i] == '<':                                        # html
            inner, i = read_balanced(text, i+1, '<', '>')
            out.append(process_html(inner))
        elif text[i] == '&':                                        # weird html
            semicolon = text.find(';', i)
            if semicolon != -1:
                entity = text[i:semicolon+1]
                try:
                    out.append(html.unescape(entity))
                except:
                    out.append(entity)
                i = semicolon + 1
            else:
                out.append(text[i])
                i += 1
            check = out.pop(len(out)-1)
            out.append(_unicode_map[check] if check in _unicode_map else check)
        elif text[i] in ("'", '"'):                                 # quotes
            if i+len(word)+3<len(text) and text[i:i+3] == "'''":
                inner, _ = read_balanced(text, i+3, "'''", "'''")
                if SequenceMatcher(None, inner, word).ratio() >= 0.85: found_word = True
            if (i > 0 and text[i-1] == text[i]) or (i < len(text)-1 and text[i+1] == text[i]):
                pass
            else:
                out.append(text[i])
            i += 1
        elif ord(text[i]) > 127:                                    # unicode
            if (text[i] in _unicode_map):
                out.append(_unicode_map[text[i]])
            else:
                out.append(text[i])                                 # but maybe keep the special things - can always delete later if necessary
            i += 1
        else:                                                       # other character - keep just in case
            out.append(text[i])
            i += 1

    result = ''.join(out)
    result = re.sub(r'[ \t]+', ' ', result)
    result = result.strip()

    if not found_word and is_example:                               # reasonable suspicion that example bad
        return ''
        #if (len(result) > 4 and all(i in ['0','1','2','3','4','5','6','7','8','9'] for i in result[0:4])) or \
        #    'http' in result:
        #    return ''
    return result

def _extract_passage(line: str) -> str:
    start = line.find('passage=') + 8 if 'passage=' in line else line.find('text=') + 5
    if start < 5:
        return None
    
    brace_count = 1
    result = []
    i = start
    while i < len(line) and brace_count > 0:
        if line[i] == '{' and i+1 < len(line) and line[i+1] == '{' or line[i] == '[' and i+1 < len(line) and line[i+1] == '[':
            brace_count += 1; result.append(line[i])
            i += 1
        elif line[i] == '}' and i+1 < len(line) and line[i+1] == '}' or line[i] == ']' and i+1 < len(line) and line[i+1] == ']':
            brace_count -= 1; result.append(line[i])
            i += 1
        elif brace_count == 1:
            if (line[i] == '|'):
                break
        result.append(line[i])
        i += 1
    if (''.join(result[-2:]) == '}}'):
        result = result[:-2]
    return ''.join(result).strip()

def clean_definition(text: str, word: str) -> str:
    text = text.lstrip("#* ").strip()
    if not text or text.startswith("{{quote"):
        return None
    output = clean_wikitext(text, word, False)
    return output.rstrip('.,;') if len(output) > 0 and output!= ":" else None

def clean_example(line: str, word: str) -> str:
    raw_text = _extract_passage(line)
    if not raw_text:
        stripped = line.lstrip('#*:').strip()
        if re.match(r'^(title|author|year|chapter|url|isbn|page|language)\s*=', stripped, re.IGNORECASE):
            return None
        if '{{' in stripped and 'passage=' not in stripped and 'text=' not in stripped and '{{...}}' not in stripped and '{{sic}}' not in stripped:
            return None
        raw_text = stripped
    return clean_wikitext(raw_text, word, True)

def get_wikitext_info(word, debug=False, max_retries=2, delay=0.5):
    """
    Gets around the finickyness of the wiktionaryparser
    """
    url = "https://en.wiktionary.org/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": word,
        "rvslots": "main",
        "rvprop": "content",
        "formatversion": "2",
        "format": "json"
    }
    headers = {
        'User-Agent': 'VOCABULATOR_app_database_scraper (supernebula31@gmail.com) PythonScript'
    }
    for attempt in range(max_retries+1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            pages = data.get("query", {}).get("pages", [])
            if not pages or "revisions" not in pages[0]:
                if debug:
                    print(f"[{word}] No revisions or page structure invalid.")
                return None
            content = pages[0]["revisions"][0]["slots"]["main"]["content"]
            parsed = wtp.parse(content)

            english_section = next((s for s in parsed.sections if s.title and "English" in s.title), None)
            if not english_section:
                return None
            outputs = []; raw_outputsDELETEME = []
            for subsection in english_section.sections:
                title = subsection.title.lower().strip() if subsection.title else ""
                if not title or title not in valid_pos: continue
                lines = subsection.string.splitlines()
                entries = []; raw_entriesDELETEME = []
                current_def = None
                for line in lines:
                    line = line.strip()
                    if debug: raw_entriesDELETEME.append(line)
                    if line.startswith("#"):                                        # new def
                        if line.startswith("#*") and current_def is not None:       # part of an example
                            ex_text = clean_example(line, word)
                            if ex_text: current_def["examples"].append(ex_text)
                        elif not line.startswith("#*"):
                            def_text = clean_definition(line, word)
                            if def_text:
                                current_def = {
                                    "definition" : def_text,
                                    "examples" : []
                                }
                                entries.append(current_def)
                        elif re.match(r'^\*\s*\{\{', line):  # probably not a real definition
                            continue
                        elif current_def is None and line:
                            def_text = clean_definition(line, word)
                            if def_text:
                                current_def = {"definition": def_text, "examples": []}
                                entries.append(current_def)
                    elif line.startswith("|passage") and current_def is not None:
                        ex_text = clean_example(line, word)
                        if ex_text: current_def["examples"].append(ex_text)


                if entries:
                    outputs.append({
                        "partOfSpeech": title,
                        "definitions": entries
                    })
                    if debug:
                        raw_outputsDELETEME.append({
                            "partOfSpeech": title,
                            "rawData": raw_entriesDELETEME
                        })

            if debug:
                return raw_outputsDELETEME, outputs
            return outputs if outputs else None

        except Exception as e:
            if attempt == max_retries:
                if debug:
                    print(f"[{word}] Failed after {max_retries+1} attempts. Error: {e}")
                return None
            if debug:
                print(f"[{word}] Attempt {attempt+1} failed. Retrying...")
            time.sleep(delay)

def generate_words_list(path, limit = 1):
    """
    generate_words_list(path, limit=1) --> None
        Populates the words.json file through a Wiktionary parser using MediaWiki API.

        path (String) - the path to the VOCABULATOR parent folder
        limit (double) = 1.0 - a limit to the number of words fetched as a % of the total dataset
                                 (so limit = .25 means only 25% of the dataset is actually parsed)
    """
    payuth = os.path.join(path, "assets", "evaluator", "WORDS.jsonl")

    #nltk.download('words')
    #englishWords = words.words()
    temp = iter_wordlist('en', wordlist='best')
    englishWords = [w for w in temp]
    englishWords.sort()

    total = len(englishWords); good=0; bad=0; wLen=0
    prev_times = [0 for _ in range(100)]

    pathgress = os.path.join(path, "assets", "evaluator", "WORDSprogress.csv")
    progress = set()
    if not os.path.isfile(pathgress):
        initialize_progress_csv(path)
    with open(pathgress, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = row['word'].lower(); d = row['defined']
            progress.add(w)
            if d=='True': 
                wLen+=len(w); good+=1
            else: bad += 1

    print('\n\n\n\n\n\n\n')
    t1 = time.time()
    for i, word in enumerate(englishWords):
        t2 = time.time(); diff = t2-t1; t1=t2; prev_times = prev_times[1:] + [diff]; avg = sum(prev_times)/len(prev_times); rem = (total-i)*avg
        print(f'\r                                                                                                                                                                                                                                                                                                                                                                                                         ', end = '')
        print(f'\r|[]|\t\t\t\t\t\t\t|[]| {word[0:2].lower():<2} | {word.lower():<30} \t\t|[]| completed: {i:06.0f}/{total} | {(i/total*100):.2f}% \t\t\t|[]| time: {(diff*1000):03.0f}ms/it | {(1/avg if avg>0 else 0):04.2f}it/s | {(rem//3600):02.0f}h:{(rem-rem//3600*3600)//60:02.0f}m:{rem-rem//3600*3600-(rem-rem//3600*3600)//60*60:02.0f}s \t\t|[]| stats: {good}\u2714 {bad}\u2716 ({good/(good+bad)*100 if (good+bad)>0 else 0:.2f}%) | \u03BC\U0001F4CF: {wLen/good if good>0 else 0:.2f} \t\t|[]|\t\t\t\t\t\t\t|[]| ', end = '')
    
        if word in progress or random.random() > limit:
            continue
        progress.add(word)
        if any(i in ['0','1','2','3','4','5','6','7','8','9','.','_'] for i in word):
            continue

        start = time.time()
        result = get_wikitext_info(word)
        elapsed = round(time.time()-start,3)            # milliseconds

        if result:
            good+=1; wLen+=len(word)
            with open(payuth, "a", encoding="utf-8") as f:
                f.write(json.dumps({word: result}, ensure_ascii=False) + "\n")
        else: bad+=1
        with open(pathgress, mode = "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([word, bool(result), elapsed])

        if elapsed > 0.7:                               # always be respectful guys
            time.sleep(0.8)
        else:
            time.sleep(random.uniform(0.25, 0.5))

    print("DONE"); return True

def json_to_jsonl(path):
    payuth = os.path.join(path, "assets", "words.json")
    with open(payuth, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(path, "assets", "thatsalottawords.jsonl"), "w", encoding="utf-8") as out:
        for word,content in data.items():
            out.write(json.dumps({word: content}) + "\n")

def generate_completed(path):
    payuth = os.path.join(path, "assets", "WORDS.jsonl")
    completedPath = os.path.join(path, "assets", "WORDScompleted.txt")
    completedWords = []
    with open(payuth, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                word = next(iter(entry))
                completedWords.append(word)
            except json.JSONDecodeError:
                continue
    for word in completedWords:
        with open(completedPath, "a", encoding="utf-8") as f:
            f.write(word + "\n")

def find_duplicates(path):
    completedPath = os.path.join(path, "assets", "WORDScompleted.txt")
    done = []
    with open(completedPath, "r", encoding = "utf-8") as f:
        for line in f:
            if line in done:
                print(line)
            else:
                done.append(line)

def initialize_progress_csv(path):
    with open(os.path.join(path, "assets", "WORDSprogress.csv"), mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'defined', 'time'])  # write header row

def reprocess_failed_examples(jsonl_path, fix_condition_fn, get_info_fn, temp_output_path=None):
    """
    Reprocess entries in a JSONL file where example conditions failed.

    Parameters:
        jsonl_path (str): Path to your original WORDS.jsonl
        fix_condition_fn (callable): Function that takes a word's data and returns True if it needs fixing
        get_info_fn (callable): Function to requery a word from Wiktionary
        temp_output_path (str): Optional path to store the fixed version before overwriting the original

    Returns:
        int: Number of entries reprocessed
    """
    temp_output_path = temp_output_path or (jsonl_path + ".tmp")
    reprocessed_count = 0
    already_checked = 0

    with open(jsonl_path, 'r', encoding='utf-8') as infile, \
         open(temp_output_path, 'w', encoding='utf-8') as outfile:
        i = 0
        for line in tqdm(infile, desc="Checking examples"):
            i += 1
            if i < already_checked:
                continue
            try:
                entry = json.loads(line)
                word = next(iter(entry))
                data = entry[word]

                if fix_condition_fn(word, data):
                    new_data = get_info_fn(word)
                    if new_data:
                        entry = {word: new_data}
                        reprocessed_count += 1
                    # else, keep the original (or optionally skip it entirely)

                outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    # Replace original with fixed version
    os.replace(temp_output_path, jsonl_path)
    return reprocessed_count

PATH = "C:\\Users\\super\\VOCABULATOR"

generate_words_list(PATH)
#initialize_progress_csv(PATH)
#find_duplicates(PATH)

#def fixed(word, data):
#    for e1 in data:
#        for e2 in e1['definitions']:
#            #for e3 in e2['examples']:
#            #    if (len(e3) > 4 and all(i in ['0','1','2','3','4','5','6','7','8','9'] for i in e3[0:4])) or ('http' in e3):
#            #        return True
#            e3 = e2['definition']
#            if (len(e3) > 4 and all(i in ['0','1','2','3','4','5','6','7','8','9'] for i in e3[0:4])) or ('http' in e3):
#                return True
#    return False   
#def fixed(word, data):
#    for e1 in data:
#        for e2 in e1['definitions']:
#            for e3 in e2['examples']:
#                for w in str(e3).split():
#                    if SequenceMatcher(None, w, word).ratio() < 0.7: return True
#    return False   
#reprocess_failed_examples(os.path.join(PATH, "assets", "WORDS.jsonl"), fixed, get_wikitext_info)