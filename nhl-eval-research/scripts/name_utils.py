import unicodedata
import re

# different spellings of a first name all map to one form
FIRST_NAME_CANON = {
    'alexander': 'alex',
    'alexandre': 'alex',
    'sasha':     'alex',
    'alexei':    'alexey',
    'cameron':   'cam',
    'christopher': 'chris',
    'daniel':    'danny',
    'egor':      'yegor',
    'evgenii':   'evgeny',
    'frederick': 'frederic',
    'freddy':    'frederic',
    'gerald':    'gerry',
    'jacob':     'jake',
    'joseph':    'joe',
    'joshua':    'josh',
    'louie':     'louis',
    'maksim':    'maxim',
    'mathew':    'matt',
    'matthew':   'matt',
    'maxime':    'max',
    'maxence':   'max',
    'maxwell':   'max',
    'mitchell':  'mitch',
    'nathan':    'nate',
    'nicholas':  'nick',
    'patrick':   'pat',
    'philip':    'phil',
    'sammy':     'sam',
    'samuel':    'sam',
    'thomas':    'tom',
    'timothy':   'tim',
    'tommy':     'tom',
    'william':   'will',
    'zachary':   'zac',
    'zach':      'zac',
    'zack':      'zac',
}

# players whose name differs a lot between sources, matched on the cleaned full name
FULL_NAME_OVERRIDES = {
    'benoit olivier groulx':  'bo groulx',
    'janis moser':            'jj moser',
    'jj moser':               'jj moser',
    'pierre olivier joseph':  'po joseph',
    'po joseph':              'po joseph',
    'viking gustafsson nyberg': 'viking gustafsson nyberg',
    # Evolving Hockey keeps the middle name, other sources drop it
    'emil martinsen lilleberg': 'emil lilleberg',
    # the 24-25 NST export turned a special character into a question mark
    'matj blumel':            'matej blumel',
}

# these two defensemen share a name with a better known forward. All Three Zones tags them
# differently, but NST lists both under the same name, so the row's position is the only way to tell
POSITION_DISAMBIGUATED_NAMES = {
    ('sebastian aho', 'D'): '5ebastian aho',
    ('elias pettersson', 'D'): 'other elias pettersson',
}


TEAM_REMAP = {
    'L.A':              'LAK',
    'N.J':              'NJD',
    'S.J':              'SJS',
    'T.B':              'TBL',
    'Utah Hockey Club': 'UTA',
    'Mammoth':          'UTA',
    'TOr':              'TOR',
    'cbJ':              'CBJ',
    'cgY':              'CGY',
    'stl':              'STL',
    'VGk':              'VGK',
}


def normalize_team(team: str) -> str:
    return TEAM_REMAP.get(team, team)


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ''

    # drop accents
    s = unicodedata.normalize('NFD', name)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')

    s = s.lower()
    s = re.sub(r'\s*\([a-z]+\)\s*$', '', s)  # trailing position tag like (d)
    s = re.sub(r'[.?]', '', s)
    s = re.sub(r'-', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()

    if s in FULL_NAME_OVERRIDES:
        return FULL_NAME_OVERRIDES[s]

    parts = s.split(' ', 1)
    if parts:
        parts[0] = FIRST_NAME_CANON.get(parts[0], parts[0])
    return ' '.join(parts)


def resolve_player_key(name: str, position: str) -> str:
    key = normalize_name(name)
    return POSITION_DISAMBIGUATED_NAMES.get((key, position), key)
