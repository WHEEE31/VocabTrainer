# VocabTrainer

An adaptive vocabulary trainer built on a 27,000-word dictionary that I scraped, cleaned, and
difficulty-rated myself. A neural network predicts how hard each word is, and a
spaced-repetition engine decides what to quiz you on next.

> **Status: work in progress.** The dataset, the difficulty model, and the learning engine
> work today from the command line. The web front end is the next milestone and has not been
> started yet.

## Why I built it

I underperformed on the vocabulary section of the SAT, and the only advice I could find was
"go read more books." I wanted something that would drill me on words at the right level and
keep bringing back the ones I kept missing, so I built it. The original, much less formal
write-up is in [`README.txt`](README.txt).

## How it works

```
 nltk word list ─► Wiktionary scraper ─► feature extraction ─► difficulty model ─► database.db ─► learner
                   definitions,          length, syllables,    residual MLP        27k words      adaptive
                   examples, synonyms    frequency, embeddings ensemble                           quizzing
```

### 1. Dataset (`src/database/rawdataset.py`)

- Starts from the `nltk` English word list and pulls each entry from the Wiktionary API.
- Parses raw wikitext into clean definitions, part-of-speech tags, example sentences, and
  synonyms. Most of the file is template, link, and HTML handling, because Wiktionary markup is
  wildly inconsistent.
- The scrape took roughly 50 hours of wall-clock time, so it is resumable and tracks per-word
  progress.

### 2. Difficulty model (`src/database/process.py`, `model*.py`)

Each word gets a difficulty score on the age-of-acquisition scale: the age at which a typical
English speaker learns the word.

- **Labels:** published age-of-acquisition ratings for about 30,000 words (Kuperman et al., 2012).
- **Features:** character count, syllable count, vowel/consonant ratio, number of definitions
  and parts of speech, corpus frequency, and a 300-dimensional word embedding reduced with PCA.
- **Models tried**, in the order I built them:

  | File | Approach |
  | --- | --- |
  | `model.py` | Feed-forward regressor, plus LightGBM and K-fold baselines |
  | `model_2.0.py` | Optuna hyperparameter search over an MLP, 5-fold × 3-model ensemble |
  | `model_3.0.py` | Residual MLP, with an option to treat difficulty as ordinal bins |

- **Results** (error in years of age-of-acquisition, from
  [`model_experiments.txt`](src/database/debugging/model_experiments.txt)):

  | Model | MAE | RMSE | Within 1.0 |
  | --- | --- | --- | --- |
  | LightGBM | 1.29 | 1.65 | 47% |
  | Single MLP | 1.24 | 1.58 | 49% |
  | K-fold MLP ensemble | 0.90 | 1.16 | 64% |

The trained model then rates every scraped word, including the ones with no published rating,
and the result is stored in `assets/database/database.db`.

### 3. Learner (`src/learner/`)

- `main.py` is a terminal quiz with three question types: pick the definition, pick the
  synonym, and mixed review. Distractors are drawn from words of similar difficulty so the
  wrong answers are plausible.
- Every answer is recorded as correct, incorrect, or "I guessed," along with response time.
  Each word moves through stages from *struggling* to *mastered*.
- `main_improved.py` replaces the first scoring heuristic with an SM-2-style spaced-repetition
  scheduler (ease factor, growing review intervals, due dates) and includes a simulated-student
  harness that plots how proficiency and intervals evolve over a study session.

## Repository layout

```
assets/
  database/
    database.db              SQLite: word, difficulty, definitions, synonyms
    DATABASE.csv             the same data as CSV
    WORDSlabeling.csv        age-of-acquisition training labels
    WORDSdifficulties.csv    model-predicted difficulty per word
src/
  database/
    rawdataset.py            Wiktionary scraper and wikitext cleaner
    process.py               feature extraction and final dataset assembly
    model.py, model_2.0.py, model_3.0.py
    experiments/             training logs, tuned hyperparameters, metrics
    debugging/               labeling rubric, experiment notes, scraper test cases
  learner/
    main.py                  terminal quiz
    main_improved.py         spaced-repetition scheduler and simulation
  frontend/
    workflow.txt             plan for the web front end
```

## Running it

```bash
pip install -r requirements.txt
python src/learner/main.py
```

Two things to know before it runs cleanly on another machine:

- The scripts currently use an absolute `ROOT` path near the top of each file. Change it to
  wherever you cloned the repository.
- `main.py` has a debug lookup and early exit just above its main loop. Remove those two lines
  to start a quiz session.

Rebuilding the dataset from scratch also needs the files listed under [Data](#data).

## Data

Large and third-party files are not committed. To rebuild everything you will need:

| File | Source |
| --- | --- |
| `assets/database/embeddings/` | [fastText](https://fasttext.cc/docs/en/crawl-vectors.html) `cc.en.300.bin` or [GloVe](https://nlp.stanford.edu/projects/glove/) 6B vectors |
| `assets/database/truth/aoa30k.xlsx` | Kuperman et al. age-of-acquisition norms |
| `assets/database/frequency/` | [FrequencyWords](https://github.com/hermitdave/FrequencyWords) English lists |

Model checkpoints (`*.pt`) and fitted scalers are also left out. The model scripts regenerate them.

## Roadmap

- [x] Wiktionary scraper and cleaned word database
- [x] Feature extraction and difficulty model
- [x] Terminal learner with adaptive question selection
- [x] Spaced-repetition scheduler prototype
- [ ] Merge the scheduler into the main learner
- [ ] Replace hard-coded paths with a config file
- [ ] Web front end: accounts, per-user progress, streaks
- [ ] Generated "use the word in a sentence" questions

## Credits

- Word list: `nltk.corpus.words`
- Definitions, examples, and synonyms: [Wiktionary](https://www.wiktionary.org/) (CC BY-SA)
- Difficulty ground truth: Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012).
  *Age-of-acquisition ratings for 30,000 English words.* Behavior Research Methods, 44, 978–990.

Built by Siddharth Nair.
