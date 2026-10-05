import random
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Import your existing code
import main  # assumes main.py is in same folder

# -------------------- CONSTANTS --------------------
EPS = 1e-9
DEFAULT_AVG_TIME = 8.0
MIN_EF = 1.3
START_EF = 2.5

# -------------------- PATCHED FUNCTIONS --------------------

def patched_get_data(self, word, allWords=False):
    # ensure columns exist
    expected_cols = ["word", "correct", "incorrect", "guessed", "proficiency",
                     "reps", "interval", "ef", "due", "avg_time"]
    for c in expected_cols:
        if c not in self.userstats.columns:
            if c != "word":
                self.userstats[c] = 0.0

    if allWords:
        return self.candidates

    # initialize row if missing
    if word not in self.userstats.index:
        tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
        default = {
            "correct": 0,
            "incorrect": 0,
            "guessed": 0,
            "proficiency": 0.0,
            "reps": 0,
            "interval": 1,
            "ef": START_EF,
            "due": tomorrow,
            "avg_time": DEFAULT_AVG_TIME
        }
        # ensure columns exist before assignment
        for k,v in default.items():
            if k not in self.userstats.columns:
                self.userstats[k] = 0.0
        self.userstats.loc[word] = default

    row = self.userstats.loc[word]
    return {
        "correct": int(row["correct"]),
        "incorrect": int(row["incorrect"]),
        "guessed": int(row["guessed"]),
        "proficiency": float(row["proficiency"]),
        "reps": int(row["reps"]),
        "interval": int(row["interval"]),
        "ef": float(row["ef"]),
        "due": row["due"],
        "avg_time": float(row["avg_time"])
    }

def patched_modify_data(self, word, stats_dict):
    # write the stats dict into the CSV-backed DataFrame and refresh candidate
    if word not in self.userstats.index:
        self.get_data(word)
    for k, v in stats_dict.items():
        self.userstats.at[word, k] = v
    # recreate candidate entry from updated stats
    self.candidates[word] = self.make_candidate(word)

def patched_make_candidate(self, word):
    info = self.get_info(word)
    stats = self.get_data(word)
    prof = stats['proficiency']
    seen = stats['correct'] + stats['incorrect'] + stats['guessed']
    error_rate = stats['incorrect'] / (seen + 1)  # +1 avoid div0

    # due calculation: positive if overdue (days overdue)
    try:
        due_date = datetime.date.fromisoformat(stats['due'])
        days_until_due = (due_date - datetime.date.today()).days
    except Exception:
        days_until_due = 0
    days_overdue = max(0, -days_until_due)
    due_factor = days_overdue / max(1, stats['interval'])

    # priority composition (tunable weights)
    # - due_factor dominant so overdue items bubble up
    # - error_rate scaled to prioritize weak items
    # - lower proficiency yields higher priority
    priority = 8.0 * due_factor + 4.0 * (error_rate * 10.0) + 3.0 * max(0, (10 - prof)) / 10.0

    # smaller new-word bonus than before so we get repeats as well
    if seen == 0:
        priority += 1.0

    # penalize words that are very mastered
    priority -= max(0, (prof - 12) * 0.35)

    priority += random.uniform(-0.3, 0.3)
    priority = max(0.01, priority)

    return {
        'word': word,
        'difficulty': info['difficulty'],
        'proficiency': prof,
        'correct': stats['correct'],
        'incorrect': stats['incorrect'],
        'guessed': stats['guessed'],
        'seen': seen,
        'reps': stats['reps'],
        'interval': stats['interval'],
        'ef': stats['ef'],
        'avg_time': stats['avg_time'],
        'due_days_overdue': days_overdue,
        'priority': priority,
        'definitions': info['definitions'],
        'synonyms': info['synonyms']
    }


def patched_process_response(self, elapsed, result, today=None):
    # consistent signature: return a stage string (as main Execution.update expects)
    if today is None:
        today = datetime.date.today()

    # fetch previous stats (copy)
    prev = self.data.get_data(self.currentWord['word'])
    prev_correct = prev['correct']
    prev_incorrect = prev['incorrect']
    prev_guessed = prev['guessed']
    prev_proficiency = prev['proficiency']

    # update counts
    if result == 'correct':
        prev_correct += 1
    elif result == 'incorrect':
        prev_incorrect += 1
    elif result == 'guessed':
        prev_guessed += 1
    seen = prev_correct + prev_incorrect + prev_guessed

    # running average of response time
    old_avg = prev['avg_time']
    new_avg = old_avg * 0.8 + elapsed * 0.2

    # map response + speed to SM-2-like grade
    if result == 'correct':
        if elapsed <= old_avg * 0.6:
            grade = 5
        elif elapsed >= old_avg * 1.5:
            grade = 3
        else:
            grade = 4
    elif result == 'guessed':
        grade = 2
    else:  # incorrect
        grade = 0

    reps, interval, ef = prev['reps'], prev['interval'], prev['ef']

    # scheduling update
    if grade < 3:
        reps, interval = 0, 1
        next_due = today + datetime.timedelta(days=1)
    else:
        reps += 1
        if reps == 1:
            interval = 1
        elif reps == 2:
            interval = 6
        else:
            # use EF to expand interval
            interval = max(1, round(interval * ef))
        # update ef using SM-2 formula variation
        ef = max(MIN_EF, ef + 0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
        next_due = today + datetime.timedelta(days=interval)

    # recompute error_rate and proficiency
    error_rate = prev_incorrect / (seen + 1)
    proficiency = min(20.0, reps * 2.0 + (1.0 - error_rate) * 6.0 + (ef - MIN_EF) * 3.0)

    stats_out = {
        'correct': prev_correct,
        'incorrect': prev_incorrect,
        'guessed': prev_guessed,
        'proficiency': round(proficiency, 3),
        'reps': reps,
        'interval': interval,
        'ef': round(ef, 3),
        'due': next_due.isoformat(),
        'avg_time': round(new_avg, 3)
    }

    # persist the updated row and update candidate
    self.data.modify_data(self.currentWord['word'], stats_out)

    # set improvement so get_next() can use it
    self.currentWord['improvement'] = stats_out['proficiency'] - (prev_proficiency or 0.0)

    # compute stage string (same logic as your thresholds)
    stage = None
    for t in sorted(self.proficiencyThreshs.keys()):
        if stats_out['proficiency'] >= t:
            stage = self.proficiencyThreshs[t]

    return stage

# -------------------- APPLY PATCH --------------------
main.Data.get_data = patched_get_data
main.Data.modify_data = patched_modify_data
main.Data.make_candidate = patched_make_candidate
main.Backend.process_response = patched_process_response

# -------------------- SIMULATION HARNESS --------------------

def simulate(backend, steps=200):
    log = []
    today = datetime.date.today()
    for step in range(steps):
        if step%20 == 0:
            today += datetime.timedelta(days=1)
        backend.today = today

        qtype, word, answer, distractors = backend.get_next()
        stats = backend.data.get_data(word)
        
        # synthetic student model: probability of correct grows with reps
        prof = stats['proficiency']
        p_correct = min(0.30 + 0.12 * stats['reps'] + 0.02 * stats['proficiency'], 0.95)

        if random.random() < p_correct:
            result = 'correct'
        else:
            result = 'guessed' if random.random() < 0.5 else 'incorrect'

        elapsed = random.uniform(3, 12)
        prof_new = backend.process_response(elapsed, result, today=today)

        cand = backend.data.make_candidate(word)
        log.append({
            'step': step,
            'day': today,
            'word': word,
            'result': result,
            'proficiency': prof,
            'reps': cand['reps'],
            'interval': cand['interval'],
            'priority': cand['priority']
        })
    return pd.DataFrame(log)

# -------------------- DASHBOARD --------------------

def plot_dashboard(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    df.groupby('step')['proficiency'].mean().plot(ax=axes[0,0], title='Avg Proficiency')
    df.groupby('step')['reps'].mean().plot(ax=axes[0,1], title='Avg Reps')
    df.groupby('step')['interval'].mean().plot(ax=axes[1,0], title='Avg Interval')
    df.groupby('step')['priority'].mean().plot(ax=axes[1,1], title='Avg Priority')
    plt.tight_layout()
    plt.show()

# -------------------- MAIN --------------------
if __name__ == "__main__":
    backend = main.Backend("assets/database/database.db", "assets/learner/user_stats.csv", "assets/learner/candidates_saved.pkl")
    df = simulate(backend, steps=300)
    plot_dashboard(df)
