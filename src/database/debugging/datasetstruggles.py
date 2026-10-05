from difflib import SequenceMatcher
import sys
import traceback
import json
import html
import os
import re
import time
import random
import requests
import wikitextparser as wtp
from tqdm import tqdm


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
                inner, _ = read_balanced(text, i+3, "'''", "'''"); print(inner)
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

def get_wikitext_info(word, debug=False):
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
    try:
        response = requests.get(url, params=params)
        data = response.json()
        content = data["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
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
        print(e)
        traceback.print_exc()
        return None
    
def is_similar(token, word, threshold=0.85):
    return SequenceMatcher(None, token, word).ratio() >= threshold

individualTesting = False

if not individualTesting:
    ##"bluebird", "bunghole", "folliculose", "orf"
    ##"construe", "lobotomy", "quixotic", "saturnine", "zeugma"
    ##"buckwheat", "aflicker", "aflutter"
    ##"barney", "binomial", "anan", "anchovy", "arrow", "bang", "bullet"
    ##"acanthopterygian", "again", "cermet", "centuried"
    
    path = os.path.join("C:\\Users\\super\\VOCABULATOR", "src", "evaluator")
    englishWords = ["acanthopterygian", "again", "cermet", "centuried"]

    pR = os.path.join(path, "testraw.json")
    pC = os.path.join(path, "testclean.json")

    rawData = {}
    cleanData = {}

    for word in tqdm(englishWords):
        raw, clean = get_wikitext_info(word, debug=True)
        rawData[word] = raw; cleanData[word] = clean

        time.sleep(random.uniform(0.5, 0.8))             # always be respectful guys

    with open(pR, "w") as f:
        json.dump(rawData, f, indent=4)
    with open(pC, "w") as f:
        json.dump(cleanData, f, indent=4)
else:
    testDefLines = [
        
    ]
    testExLines = [
        "#* {{RQ:Smollett Peregrine Pickle|chapter=He Introduces His New Friends to Mr. Jolter, with whom the Doctor Enters into a Dispute upon Government, which had Well Nigh Terminated in Open War|page=127|passage=The\u017fe gentlemen, with an equal \u017fhare of pride, pedantry, and '''\u017faturnine''' di\u017fpo\u017fition, were by the accidents of education and company, diametrically oppo\u017fite in political maxims; [...]}}",
        "|passage=Thus, in a sentence such as:<br> (113) &nbsp;&nbsp;&nbsp;&nbsp; John considers [<sub>S</sub> ''Fred'' to be too sure of ''himself'']<br> the italicised Reflexive ''himself'' can only be '''construed''' with ''Fred'', not with ''John'': this follows from our assumption that non-subject Reflexives must have an antecedent within their own S. Notice, however, that in a sentence such as:<br> (114) &nbsp;&nbsp;&nbsp;&nbsp; ''John'' seems to me [<sub>S</sub> \u2014 to have perjured ''himself'']<br> ''himself'' must be '''construed''' with ''John''.}}",
        "#* {{quote-journal|en|date=1991-11-23|author=D. J. Nokes; R. M. Anderson|title=Vaccine safety versus vaccine efficacy in mass immunisation programmes|journal=w:The Lancet|volume=338|number=8778|doi=10.1016/0140-6736(91)92601-W|page=1309|passage=Assuming a normal approximation to '''binomial''' probabilities the proportion of total complications reported for 1979\u201385 in the age class 0\u201314 years was significantly higher than the proportion in the same age class for the period 1962\u201369 (p < 0\u00b70001)}}"
    ]
    
    #print("\n------definition testing------\n")
    #for line in testDefLines:
    #    print(repr(clean_definition(line)))
    #print("\n------example testing------\n")
    #for line in testExLines:
    #    print(repr(clean_example(line)))

    print(is_similar("a-flick-", "aflicker"))










'''


Hi, Please think deeply about how to do this in a way that is as robust as possible (does not need extra tuning for all the little edge cases)

I want to be able to take lines scraped from a wiktionary website and clean them up. I have already written code that identifies the lines that need to be cleaned, but I need code that can do the cleaning. This code must be as straightforward as possible, in that there is next to no need for extra hardcoding for edge cases. I will attach two files. One file is the raw information that I get directly from the website. It has a lot of lines, because the proper lines haven't been selected yet. The other file has much less lines. This file contains the selected lines that I want for each word, as well as the target of the cleaning operation. I suggest that you create two python functions: one for the definitions, and one for the examples. This is because the way these two different types of information are formatted on the website are different and thus different cleaning methodologies are necessary for each type.

Pay close attention to what I want to keep and what I want to discard for each line. Especially, the punctuation. And see how some of the punctuation like ' is kept, but others, like '' dfsdf '' are discarded. 

Also look at the unicode stuff - most of them are converted into normal ascii, like \017f, which becomes s. Some rules about that:
 - \u017f = s 
 - \u2019 = '
 - \u2014 = -
 - \u201c, \u201d = ", "
 - \u00e6 = ae

Finally, you see how I deleted the ''' ''' surrounding the word in the examples, but if that's really triccky/tedious/annoying to implement then feel free to keep those in. It's not that big of a deal. The other stuff is way more important to deal with than this.

'''