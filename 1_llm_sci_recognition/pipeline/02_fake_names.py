#!/usr/bin/env python3
"""Step 2: 1,000 fabricated physicist names for the false-positive probe.

Source: query_recognition_dimensions_multifields.ipynb, cell 24, first half
("GENERATE FAKE NAMES VIA CHARACTER PERTURBATION"). Logic verbatim.

Each fake keeps a real physicist's first name and perturbs the surname once:
vowel substitution, consonant doubling/un-doubling, adjacent swap or interior
deletion, drawn with weights 3:2:1:1 (Latin-script surnames of >= 3 characters;
for hyphenated surnames one part is perturbed). A candidate is rejected if
(last, first) is any author in the full panel, if the new surname is any real
surname in the panel, or if it repeats an earlier fake. RANDOM_STATE = 2024.

Step 3 reads the list (first and last name only) as inputs/fake_names_physics.csv.
The list is not shipped: each name is a one-letter edit of a real author's surname.

Input   PANEL_CSV (Dimensions author panel; licensed, not shipped)
Output  MULTI_DIR/fake_names_physics_perturbed.csv
        (author_first_name, author_last_name, _source_last; keep private:
        _source_last is the real surname)
"""
import os
import re

import numpy as np
import pandas as pd

from paths import MULTI_DIR, PANEL_CSV

N_FAKE = 1000
PHYSICS_FIELD = 'Physical Sciences'
RANDOM_STATE = 2024
PROBE_DIR = MULTI_DIR

VOWELS = set('aeiouAEIOU')


# ---- Perturbation engine ----------------------------------------------------

def _is_latin_name(name):
    """Check that a name is Latin-script and long enough to perturb."""
    s = str(name).strip()
    if len(s) < 3:
        return False
    # Allow letters, hyphens, spaces, apostrophes
    return bool(re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ' -]+$", s))


def _perturb_part(s, rng):
    """Perturb a single name part (no hyphens). Returns None on failure."""
    chars = list(s)
    if len(chars) < 3:
        return None

    # Weighted draw: vowel_sub is most natural
    op = rng.choice(['vowel_sub', 'vowel_sub', 'vowel_sub',
                     'double', 'double',
                     'swap', 'delete'])

    if op == 'vowel_sub':
        vpos = [i for i, c in enumerate(chars) if c in VOWELS and i > 0]
        if not vpos:
            return None
        i = rng.choice(vpos)
        alts = [v for v in 'aeiou' if v != chars[i].lower()]
        new = rng.choice(alts)
        chars[i] = new.upper() if chars[i].isupper() else new

    elif op == 'double':
        # Find a consonant to double, or un-double a pair
        consonant_pos = [i for i in range(1, len(chars))
                         if chars[i] not in VOWELS and chars[i].isalpha()]
        if not consonant_pos:
            return None
        i = rng.choice(consonant_pos)
        if i + 1 < len(chars) and chars[i].lower() == chars[i + 1].lower():
            chars.pop(i + 1)             # un-double: "ll" -> "l"
        else:
            chars.insert(i + 1, chars[i].lower())  # double: "l" -> "ll"

    elif op == 'swap':
        positions = list(range(1, len(chars) - 1))
        if not positions:
            return None
        i = rng.choice(positions)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]

    elif op == 'delete':
        positions = list(range(1, len(chars) - 1))
        if not positions:
            return None
        i = rng.choice(positions)
        chars.pop(i)

    result = ''.join(chars)
    return result if result != s and len(result) >= 2 else None


def perturb_last_name(last, rng, max_tries=15):
    """Perturb a last name. Hyphenated surnames: perturb one random part."""
    if not _is_latin_name(last):
        return None

    for _ in range(max_tries):
        if '-' in last:
            parts = last.split('-')
            idx = rng.randint(0, len(parts))
            new_part = _perturb_part(parts[idx], rng)
            if new_part is None:
                continue
            parts[idx] = new_part
            return '-'.join(parts)
        else:
            result = _perturb_part(last, rng)
            if result is not None:
                return result
    return None


def main():
    dimensions = pd.read_csv(PANEL_CSV)

    # ---- Build the set of all real (last, first) pairs ---------------------
    phys = dimensions[dimensions['modal_field_name'] == PHYSICS_FIELD].copy()
    phys = phys[phys['author_first_name'].notna()
                & phys['author_last_name'].notna()].copy()
    print(f"Physics pool: {len(phys):,} scientists")

    # Check against ALL scientists, not just physics
    real_combos = set(
        zip(dimensions['author_last_name'].astype(str),
            dimensions['author_first_name'].astype(str))
    )
    # Also build a set of all known last names (for a secondary guard)
    all_real_lasts = set(dimensions['author_last_name'].astype(str).unique())
    print(f"Real name pairs to avoid: {len(real_combos):,}")
    print(f"Known last names: {len(all_real_lasts):,}")

    # ---- Generate ----------------------------------------------------------
    rng = np.random.RandomState(RANDOM_STATE)

    # Shuffle the physics pool so we draw from diverse names
    pool = phys.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    fakes = []
    seen = set()
    pool_idx = 0

    while len(fakes) < N_FAKE and pool_idx < len(pool) * 3:
        row = pool.iloc[pool_idx % len(pool)]
        pool_idx += 1

        first = str(row['author_first_name']).strip()
        last = str(row['author_last_name']).strip()

        new_last = perturb_last_name(last, rng)
        if new_last is None:
            continue

        combo = (new_last, first)
        if combo in real_combos or combo in seen:
            continue

        # Secondary guard: perturbed surname must not be any real surname
        if new_last in all_real_lasts:
            continue

        seen.add(combo)
        fakes.append({
            'author_first_name': first,
            'author_last_name':  new_last,
            '_source_last':      last,
        })

    fake_df = pd.DataFrame(fakes)
    print(f"\nGenerated {len(fake_df):,} fake names from {pool_idx:,} attempts")

    os.makedirs(PROBE_DIR, exist_ok=True)
    fake_df.to_csv(os.path.join(PROBE_DIR, 'fake_names_physics_perturbed.csv'),
                   index=False)

    n_real_last = fake_df['author_last_name'].isin(all_real_lasts).sum()
    print(f"Perturbed last names that match a known real surname: {n_real_last} "
          f"(should be 0 — filtered out)")


if __name__ == "__main__":
    main()
