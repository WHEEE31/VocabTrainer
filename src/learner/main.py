import os
import sys
import json
import time
import tqdm
import pickle
import string
import pandas
import random
import sqlite3
import keyboard
import threading

class Data:
    def __init__(self, path_to_database, path_to_userstats, path_to_pickle):
        self.conn = sqlite3.connect(path_to_database)
        self.cursor = self.conn.cursor()

        if not os.path.exists(path_to_userstats):
            initial_data = pandas.DataFrame(columns=["word", "correct", "incorrect", "guessed", "proficiency"])
            initial_data.to_csv(path_to_userstats, index=False)

        # Read the CSV and set index
        self.userstats = pandas.read_csv(path_to_userstats)
        self.userstats.set_index("word", inplace=True)
        self.statsPath = path_to_userstats

        self.candidates = {}
        if os.path.exists(path_to_pickle):
            with open(path_to_pickle, "rb") as f:
                self.candidates = pickle.load(f)
        else:
            rows = self.get_words()
            for w in tqdm.tqdm(rows, desc='Loading...'):
                self.candidates[w] = self.make_candidate(w)
        self.candidatesPath = path_to_pickle

    def get_words(self):
        self.cursor.execute("SELECT word FROM words")
        return [row[0] for row in self.cursor.fetchall()]

    def get_info(self, word):
        #print(f'get_info({word})', end ='\r')
        self.cursor.execute("SELECT * FROM words WHERE word = ?", (word,))
        row = self.cursor.fetchone()
        if row:
            word, difficulty, definitions, synonyms = row
            return {'difficulty':difficulty, 'definitions':[f'({d[0]}) {d[1]}' for d in json.loads(definitions)], 'synonyms':json.loads(synonyms)}
        else:
            return None
    
    def get_data(self, word, allWords=False):
        #print(f'get_data({word}, allWords={allWords})', end = '\r')
        if allWords: return self.candidates
        if word not in self.userstats.index:
            self.userstats.loc[word] = {
                "correct": 0,
                "incorrect": 0,
                "guessed": 0,
                "proficiency": 0.00
            }
        return {'correct' : self.userstats.loc[word]['correct'],
                'incorrect' : self.userstats.loc[word]['incorrect'],
                'guessed' : self.userstats.loc[word]['guessed'],
                'proficiency' : self.userstats.loc[word]['proficiency']}
    
    def modify_data(self, word, correct, incorrect, guessed, proficiency):
        self.userstats.loc[word] = [correct, incorrect, guessed, proficiency]
        self.candidates[word] = self.make_candidate(word)

    def save_data(self):
        self.userstats.to_csv(self.statsPath, index=True)

        with open(self.candidatesPath, "wb") as f:
            pickle.dump(self.candidates, f)

    def make_candidate(self, word):
        info = self.get_info(word)
        stats = self.get_data(word)
        proficiency = stats.get('proficiency', 0) or 0
        correct = stats.get('correct', 0) or 0
        incorrect = stats.get('incorrect', 0) or 0
        guessed = stats.get('guessed', 0) or 0
        seen = correct + incorrect + guessed

        base_priority = max(0.1, (20 - proficiency) + (incorrect * 2) - correct + (seen * 0.5))
        if seen == 0:
            base_priority += 3.0
        base_priority += random.uniform(-1.0, 1.0)

        return {
            'word': word,
            'difficulty': info['difficulty'],
            'proficiency': proficiency,
            'correct': correct,
            'incorrect': incorrect,
            'guessed':guessed,
            'seen': seen,
            'priority': base_priority,
            'definitions': info['definitions'],
            'synonyms': info['synonyms']
        }


class Backend:
    def __init__(self, path_to_database, path_to_userstats, path_to_pickle):
        self.data = Data(path_to_database, path_to_userstats, path_to_pickle)

        self.currentWord = {'word' : '', 'improvement': 0}
        self.proficiencyWeights = {'correct %' : 1,
                                   'incorrect %' : -1,
                                   'guessed %' : 0,
                                   'seen' : 0.25,
                                   'correct speed' : {'start' : 1, 'operation' : '/', 'multiplier' : 0.25},
                                   'incorrect speed' : {'start' : 0, 'operation' : '-', 'multiplier' : 0.25},
                                   'guessed speed' : {'start' : 1, 'operation' : '/', 'multiplier' : 4}}
        self.proficiencyThreshs = {float('-inf') : 'struggling',
                                   0 : 'recognized',
                                   5 : 'acquainted',
                                   10 : 'proficient',
                                   15 : 'internalized',
                                   20 : 'mastered'}
        
        self.questionTypes = ['definition', 'synonym', 'review']

    def get_next(self):
        t1 = time.perf_counter()
        #print(f'1. [{time.perf_counter()-t1}]') # 1. Load candidates list (words + difficulty)

        #print(f'2. [{time.perf_counter()-t1}]') # 2. Cache word info if not already cached
        if not hasattr(self, '_word_info_cache'):
            self._word_info_cache = {}

            # Precompute distractor pools
            self._defs_by_diff = {}
            self._syns_by_diff = {}
            words = self.data.get_words()
            for w in words:
                data = self.data.get_info(w)
                diff = data['difficulty']
                self._defs_by_diff.setdefault(diff, []).extend(data['definitions'])
                self._syns_by_diff.setdefault(diff, []).extend(data['synonyms'])

        #print(f'3. [{time.perf_counter()-t1}]') # 3. Always fetch fresh stats for each word
        candidates = list(self.data.get_data('',allWords=True).values())

        #print(f'4. [{time.perf_counter()-t1}]') # 4. Difficulty targeting
        diff_shift = 0
        imp = self.currentWord.get('improvement', 0) or 0
        if imp > 0.5:
            diff_shift = 1
        elif imp < -0.5:
            diff_shift = -1
        avg_diff = sum(c['difficulty'] for c in candidates) / len(candidates)
        target_diff = round(avg_diff + diff_shift)

        filtered = [c for c in candidates if abs(c['difficulty'] - target_diff) <= 2]
        pool = filtered if filtered else candidates

        #print(f'5. [{time.perf_counter()-t1}]') # 5. Weighted random pick
        total = sum(max(0.01, c['priority']) for c in pool)
        pick = random.uniform(0, total)
        upto = 0
        chosen = pool[-1]
        for c in pool:
            upto += max(0.01, c['priority'])
            if pick <= upto:
                chosen = c
                break

        word = chosen['word']
        self.currentWord['word'] = word

        #print(f'6. [{time.perf_counter()-t1}]') # 6. Distractor helpers (in-memory)
        def other_definitions(n, exclude, diff):
            defs = [d for d in self._defs_by_diff.get(diff, []) if not d.startswith(f"({exclude}")]
            if len(defs) < n:
                defs = sum(self._defs_by_diff.values(), [])
            return random.sample(defs, n) if defs else ["an unrelated phrase"] * n

        def other_synonyms(n, exclude, diff):
            syns = [s for s in self._syns_by_diff.get(diff, []) if s != exclude]
            if len(syns) < n:
                syns = sum(self._syns_by_diff.values(), [])
            return random.sample(syns, n) if syns else ["unrelated"] * n

        #print(f'7. [{time.perf_counter()-t1}]') # 7. Question type selection
        prof = chosen['proficiency']
        review_prob = min(0.5, 0.05 + (prof / 40.0) + (chosen['seen'] * 0.02))
        if random.random() < review_prob:
            questionType = 'review'
        else:
            if prof < 5:
                questionType = 'definition'
            elif prof < 12:
                questionType = random.choices(['definition', 'synonym'], weights=[0.6, 0.4])[0]
            else:
                questionType = 'synonym'

        #print(f'8. [{time.perf_counter()-t1}]') # 8. Return question
        if questionType == 'definition':
            correct_def = random.choice(chosen['definitions']) if chosen['definitions'] else f"Definition of {word}"
            distractors = other_definitions(3, exclude=word, diff=chosen['difficulty'])
            return 'definition', word, correct_def, distractors

        elif questionType == 'synonym':
            correct_syn = random.choice(chosen['synonyms']) if chosen['synonyms'] else word
            distractors = other_synonyms(3, exclude=word, diff=chosen['difficulty'])
            return 'synonym', word, correct_syn, distractors

        else:  # review
            def_ans = random.choice(chosen['definitions']) if chosen['definitions'] else f"Definition of {word}"
            syn_ans = random.choice(chosen['synonyms']) if chosen['synonyms'] else word
            distractors = other_synonyms(3, exclude=word, diff=chosen['difficulty']) + \
                        other_definitions(3, exclude=word, diff=chosen['difficulty'])
            return 'review', word, (def_ans, syn_ans), distractors

    def process_response(self, elapsed, result):
        prevData = self.data.get_data(self.currentWord['word'])
        prevData[result] += 1
        (c, i, g) = list(prevData.values())[:3]
        p =       round(prevData['correct'] / sum(list(prevData.values())[:3]) * self.proficiencyWeights['correct %'] + \
                        prevData['incorrect'] / sum(list(prevData.values())[:3]) * self.proficiencyWeights['incorrect %'] + \
                        prevData['guessed'] / sum(list(prevData.values())[:3]) * self.proficiencyWeights['guessed %'] + \
                        sum(list(prevData.values())[:3]) * self.proficiencyWeights['seen'] + \
                        eval(f'{self.proficiencyWeights[result+' speed']['start']} \
                             {self.proficiencyWeights[result+' speed']['operation']} \
                                ({elapsed}* \
                                    {self.proficiencyWeights[result+' speed']['multiplier']})'), 3)
        
        self.data.modify_data(self.currentWord['word'], c, i, g, p)
        self.currentWord['improvement'] = p - prevData['proficiency']

        for t in self.proficiencyThreshs.keys():
            if p >= t:
                stage = self.proficiencyThreshs[t]
        
        return stage
    
    def cache(self):
        self.data.save_data()

class Frontend:
    def __init__(self):
        self.tabsize = 4
        self.choiceLabels = [f'{i}. ' for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']]
        self.guessText = ['No idea', 'I don\'t know', 'Beats me', 'Puzzled', 'This is a guess', 'HUH?!']
        self.correctText = ['Great job!', 'Way to go!', 'Well done!', 'Nice work!']
        self.incorrectText = ['Sorry, that isn\'t correct', 'Oops!', 'Womp womp.', 'Bruh', 'L bozo']
        self.prompter = ' '*self.tabsize + '>>>'

        self.outputFlags = {0: "correct", 1: "incorrect", 2: "guessed"}

    def question(self, prompt, choices, correct, num=None):     # the last option is always 'guess'
        #print(f'Prompt: {prompt}; Choices: {choices}; Correct: {correct}')
        def normalize_choice(text):
            if text is None:
                return None
            t = text.strip().lower()
            t = t.rstrip(string.punctuation + " ")
            if t.isalpha() and len(t) == 1:
                return ord(t) - ord('a')
            if t.isdigit():
                return int(t) - 1
            return t

        first = True
        t1 = time.perf_counter()

        if num: print(f'Question {num}:', end='')
        while True:
            print(f'{prompt}')
            
            for (text, label) in zip(choices+[random.choice(self.guessText)], self.choiceLabels):
                print(' '*self.tabsize + label + text)
            
            response = input(self.prompter)
            if response.strip().lower() == 'q':
                return
            if first:
                t2 = time.perf_counter()
                elapsed = t2-t1
            first = False

            response = normalize_choice(response)
            for i, c in enumerate(self.choiceLabels):
                if response == normalize_choice(c):
                    if i == correct:
                        print(random.choice(self.correctText))
                        return elapsed, self.outputFlags[0]
                    elif i == len(choices):
                        print('The correct answer was ' + self.choiceLabels[correct] + choices[correct])
                        return elapsed, self.outputFlags[2]
                    else:
                        print(random.choice(self.incorrectText))
                        print('The correct answer was ' + self.choiceLabels[correct] + choices[correct])
                        return elapsed, self.outputFlags[1]
                if i >= len(choices): break
                    
            print(f"Please respond with one of {self.choiceLabels[:len(choices)+1]}")
        
    def defQ(self, word, definition, distractors):
        choices = [definition] + distractors
        random.shuffle(choices)
        correct = choices.index(definition)

        return self.question(f'What is the definition of "{word}" ?',
                             choices,
                             correct)

    def synQ(self, word, synonym, distractors):
        choices = [synonym] + distractors
        random.shuffle(choices)
        correct = choices.index(synonym)

        return self.question(f'Which is the best synonym to "{word}" ?',
                             choices,
                             correct)

    def reviewQ(self, word, definition, synonym, distractors):
        print('Review question!!!!')

        k = random.random()
        if k<0.5:
            return self.defQ(word, definition, distractors)
        else:
            return self.synQ(word, synonym, distractors)

class Execution:
    def __init__(self, frontend, backend):
        self.frontend = frontend
        self.backend = backend

    def update(self):
        # get next question
        questionType, word, answer, distractors = self.backend.get_next()

        # ask the question
        try:
            if questionType == 'definition': elapsed, result = self.frontend.defQ(word, answer, distractors)
            elif questionType == 'synonym': elapsed, result = self.frontend.synQ(word, answer, distractors)
            elif questionType == 'review': elapsed, result = self.frontend.reviewQ(word, answer[0], answer[1], distractors)
        except TypeError as e:
            print(e)
            self.quit()

        # process the answer
        stage = self.backend.process_response(elapsed, result)
        print(f'You\'re at stage "{stage}" for word "{word}"')

        time.sleep(1.5)
        inp = input('[enter] to continue, [q] to quit')
        if inp.strip().lower() == 'q':
            self.quit()
    
    def quit(self):
        print('\n-------- Saving your progress --------')
        self.backend.cache()
        sys.exit(0)

#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

if __name__ == "__main__":
    ROOT = "C:\\Users\\super\\Documents\\projects\\VOCABULATOR"

    frontend = Frontend()
    backend = Backend(os.path.join(ROOT, "assets", "database", "database.db"),
                os.path.join(ROOT, "assets", "learner", "user_stats.csv"),
                os.path.join(ROOT, "assets", "learner", "candidates_saved.pkl"))
    run = Execution(frontend, backend)

    #'''
    print(backend.data.get_info('parochial')['difficulty'])
    sys.exit(0)

    '''
    annul : 12.5
    anguish : 11.54
    '''
    #'''
    
    while True:
        run.update()