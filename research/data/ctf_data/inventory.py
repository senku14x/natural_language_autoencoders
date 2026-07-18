"""Candidate value inventories and the fixed neutral preamble.

These are CANDIDATE lists — the tokenizer audit (audit.py) filters them to
values that are a single token in leading-space form. Nothing downstream may
use a value that did not survive the audit.
"""

# Colors: candidate answer values for stratum S.
COLOR_CANDIDATES = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
    "white", "brown", "gray", "grey", "gold", "silver", "violet", "indigo",
    "cyan", "magenta", "teal", "maroon", "navy", "olive", "beige", "tan",
    "crimson", "scarlet", "azure", "turquoise", "lavender", "coral", "salmon",
    "amber", "ivory", "khaki", "lime", "mint", "rose", "ruby", "emerald",
    "sapphire", "bronze", "copper", "charcoal", "cream", "peach", "plum",
    "rust", "slate", "aqua", "burgundy", "mustard", "chartreuse",
]

# Common first names: candidate answer values for stratum N (name queries).
NAME_CANDIDATES = [
    "Alice", "Bob", "John", "Mary", "James", "Emma", "Sarah", "David",
    "Michael", "Anna", "Tom", "Jack", "Maria", "Priya", "Wei", "Ahmed",
    "Fatima", "Carlos", "Sofia", "Elena", "Ivan", "Yuki", "Kenji", "Omar",
    "Layla", "Ravi", "Diego", "Lucia", "Marco", "Giulia", "Hans", "Greta",
    "Pierre", "Claire", "Chen", "Ming", "Aisha", "Nia", "Samir", "Noor",
    "Arjun", "Rohan", "Meera", "Vikram", "Isha", "Olga", "Pavel", "Nina",
    "Sven", "Ingrid", "Mateo", "Camila", "Andres", "Kofi", "Amina", "Yusuf",
    "Hassan", "Mariam", "Jamal", "Marcus", "Jasmine", "Malik", "Andre",
    "Anne", "Jane", "Paul", "Mark", "Luke", "Peter", "Susan", "Karen",
    "Linda", "Nancy", "Helen", "Laura", "Kevin", "Brian", "Eric", "Alan",
    "Ryan", "Adam", "Sam", "Ben", "Dan", "Joe", "Amy", "Kate", "Lucy",
    "Rachel", "Hannah", "Grace", "Chloe", "Sophie", "Oliver", "Henry", "Leo",
    "Max", "Felix", "Oscar", "Hugo", "Victor", "Simon", "Martin", "Robert",
    "Richard", "Charles", "George", "Edward", "Frank", "Albert", "Arthur",
    "Walter", "Harold", "Ralph", "Eugene", "Howard", "Zara", "Tariq",
    "Leila", "Kavya", "Ananya", "Dmitri", "Astrid", "Henrik",
]

# Cities: context values / distractor-query answers for stratum N.
CITY_CANDIDATES = [
    "Paris", "London", "Tokyo", "Berlin", "Madrid", "Rome", "Moscow",
    "Beijing", "Sydney", "Toronto", "Chicago", "Boston", "Denver", "Austin",
    "Dublin", "Vienna", "Prague", "Cairo", "Delhi", "Mumbai", "Seattle",
    "Portland", "Atlanta", "Houston", "Dallas", "Phoenix", "Miami",
    "Detroit", "Montreal", "Vancouver", "Ottawa", "Melbourne", "Brisbane",
    "Perth", "Auckland", "Oslo", "Stockholm", "Helsinki", "Copenhagen",
    "Amsterdam", "Brussels", "Geneva", "Zurich", "Munich", "Hamburg",
    "Frankfurt", "Warsaw", "Budapest", "Athens", "Lisbon", "Barcelona",
    "Valencia", "Seville", "Milan", "Naples", "Turin", "Florence", "Venice",
    "Osaka", "Kyoto", "Seoul", "Shanghai", "Singapore", "Bangkok", "Jakarta",
    "Manila", "Nairobi", "Lagos", "Accra", "Istanbul", "Ankara", "Tehran",
    "Riyadh", "Dubai", "Karachi", "Lahore", "Dhaka", "Chennai", "Kolkata",
    "Bangalore", "Hyderabad", "Lima", "Bogota", "Santiago", "Havana",
]

# Nonce entity words for stratum S. Not answers — they need not be
# single-token; they only need identical tokenization within a pair, which
# is automatic (the entity string is identical across pair members) and is
# still asserted at the prompt level (exactly-one-differing-token invariant).
NONCE_ENTITIES = [
    "dax", "wug", "blicket", "fep", "tob", "zorp", "glorp", "snib", "trell",
    "plif", "vimp", "gub", "mib", "lorp", "florp", "zim", "quab", "norg",
    "pib", "waff", "jeck", "blorf", "sprock", "tazz",
]

# Fixed neutral preamble (PROVISIONAL — flagged for user sign-off; see
# research/data/README.md). Prepended IDENTICALLY to both pair members in the
# preamble=True variant, to push the read site past token position 50
# (released AV's stage-0 _MIN_POSITION). Constraints: constant across every
# example and condition; no overlap with any color/name/city/nonce candidate;
# no numerals; register close to generic web prose.
PREAMBLE_TEXT = (
    "This page is part of a general reading exercise collected for a records "
    "archive. The passages in this archive are short and self contained, and "
    "each one is followed by a single question about its contents. Readers "
    "are asked to rely only on the statements given in the passage itself, "
    "and to keep every reply as brief as possible, using one word whenever "
    "that is enough.\n\n"
)

# Words whose presence in the preamble would collide with an answer value.
# Checked by audit.check_preamble_clean().
def preamble_collisions(preamble: str, candidates: list[str]) -> list[str]:
    lowered = preamble.lower()
    hits = []
    for value in candidates:
        # word-boundary-ish containment check; conservative (substring)
        if value.lower() in lowered.split() or f" {value.lower()} " in lowered:
            hits.append(value)
    return hits
