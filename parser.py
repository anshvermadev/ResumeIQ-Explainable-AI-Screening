# ================================================================
# parser.py — FINAL VERSION WITH NLTK
# Rule-based NLP resume parser
# NLTK used for:
#   1. Lemmatization — catches word variants (certified/certifying)
#   2. Stopword filtering — accurate resume word count
#   3. POS tagging — soft skills detection via adjective/noun tags
#   4. Tokenization — accurate word splitting
# ================================================================

import re
import pdfplumber
import docx
import nltk

# Download required NLTK data (only runs once)
for pkg in ['punkt', 'averaged_perceptron_tagger', 'stopwords',
            'wordnet', 'maxent_ne_chunker', 'words', 'punkt_tab',
            'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.stem     import WordNetLemmatizer
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk          import pos_tag

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))


# ----------------------------------------------------------------
# KEYWORD DICTIONARIES
# ----------------------------------------------------------------

PROGRAMMING_LANGUAGES = [
    'python', 'java', 'javascript', 'typescript', 'c++', 'cpp',
    'c#', 'csharp', 'ruby', 'go', 'golang', 'rust', 'swift',
    'kotlin', 'scala', 'matlab', 'php', 'dart', 'perl', 'haskell',
    'lua', 'shell', 'bash', 'sql', 'html', 'css', 'react', 'node',
    'nodejs', 'vue', 'angular', 'godot', 'gdscript', 'r',
]

SOFT_SKILL_KEYWORDS = [
    'communication', 'leadership', 'teamwork', 'team work', 'team player',
    'problem solving', 'problem-solving', 'problem - solving',
    'critical thinking', 'time management', 'adaptability', 'creativity',
    'collaboration', 'interpersonal', 'presentation', 'negotiation',
    'conflict resolution', 'decision making', 'decision-making',
    'emotional intelligence', 'mentoring', 'coaching',
    'public speaking', 'active listening', 'attention to detail',
    'work ethic', 'self motivated', 'self-motivated',
    'organised', 'organized', 'multitasking', 'multi-tasking',
]

SOFT_SKILL_LEMMAS = set()
for kw in SOFT_SKILL_KEYWORDS:
    for word in kw.split():
        SOFT_SKILL_LEMMAS.add(LEMMATIZER.lemmatize(word.lower()))

SOFT_SKILL_POS = {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'VBG', 'VBN'}

CERTIFICATION_KEYWORDS = [
    'certified', 'certification', 'certificate', 'certify',
    'coursera', 'udemy', 'edx', 'nptel', 'cisco',
    'comptia', 'pmp', 'agile', 'scrum',
    'oracle', 'microsoft certified', 'google cloud certified',
    'aws certified', 'azure certified', 'fdp',
]

CERTIFICATION_LEMMAS = set(
    LEMMATIZER.lemmatize(kw.split()[0].lower())
    for kw in CERTIFICATION_KEYWORDS
)

INTERNSHIP_ROLE_PATTERNS = [
    r'\bintern\b(?!\s*connect)',
    r'\binternship\b',
    r'\btrainee\b',
    r'\bapprentice\b',
    r'summer\s+intern',
    r'winter\s+intern',
    r'industrial\s+training',
    r'research\s+intern\b',
    r'software\s+engineering\s+intern',
    r'engineering\s+intern\b',
]

HACKATHON_KEYWORDS = [
    'hackathon', 'datathon', 'kaggle competition',
    'code competition', 'challenge winner', 'finalist',
    'code.fun.do', 'shaastra', 'techfest',
]

RESEARCH_KEYWORDS = [
    'research paper', 'published', 'publication', 'journal paper',
    'conference paper', 'arxiv', 'ieee', 'acm ', 'springer',
    'thesis', 'dissertation', 'patent',
]

EDUCATION_KEYWORDS = {
    'PhD'      : ['phd', 'ph.d', 'doctor of philosophy', 'doctorate'],
    'Masters'  : ['master', 'm.tech', 'mtech', 'm.e.', 'mba',
                  'm.sc', 'ms ', 'm.s.', 'post graduate',
                  'postgraduate', 'dual degree'],
    'Bachelors': ['bachelor', 'b.tech', 'btech', 'b.e.', 'be ',
                  'b.sc', 'bsc', 'undergraduate', 'b.com', 'bca', 'bba'],
}

BACHELORS_KEYWORDS = [
    'b.tech', 'btech', 'b.e.', 'b.e ', 'be ',
    'b.sc', 'bsc', 'b.com', 'bcom',
    'bca', 'bba', 'bachelor', 'undergraduate',
    'b.arch', 'b.pharma', 'mbbs',
]

TIER1_UNIVERSITIES = [
    'iit ', 'iit,', 'iitm', 'iitb', 'iitd', 'iitk', 'iitr', 'iitkgp', 'iitg',
    'indian institute of technology',
    'iisc', 'indian institute of science',
    'bits pilani', 'bits goa', 'bits hyderabad',
    'aiims', 'all india institute of medical sciences',
    'iim ', 'iima', 'iimb', 'iimc', 'iim lucknow',
    'indian institute of management',
    'isi kolkata', 'indian statistical institute',
    'tifr', 'tata institute of fundamental research',
    'iiit hyderabad', 'iiit bangalore',
    'delhi university', 'du ',
    'jawaharlal nehru university', 'jnu',
    'banaras hindu university', 'bhu',
    'jamia millia islamia',
    'ashoka university',
    'shiv nadar university',
    'mit', 'massachusetts institute of technology',
    'stanford', 'stanford university',
    'harvard', 'harvard university',
    'caltech', 'california institute of technology',
    'uc berkeley', 'university of california berkeley',
    'ucla',
    'carnegie mellon',
    'princeton',
    'yale',
    'columbia university',
    'university of chicago',
    'oxford', 'university of oxford',
    'cambridge', 'university of cambridge',
    'imperial college london',
    'ucl', 'university college london',
    'london school of economics', 'lse',
    'eth zurich',
    'epfl',
    'tum', 'technical university of munich',
    'university of amsterdam',
    'sorbonne university',
    'national university of singapore', 'nus',
    'nanyang technological university', 'ntu singapore',
    'tsinghua university',
    'peking university',
    'university of hong kong',
    'university of toronto',
    'university of melbourne',
    'university of sydney'
]

TIER2_UNIVERSITIES = [
    'vit ', 'vit vellore',
    'srm ', 'srm university',
    'manipal', 'manipal university',
    'amity',
    'symbiosis',
    'christ university',
    'anna university',
    'pune university',
    'osmania university',
    'jadavpur university',
    'psg college',
    'bharati vidyapeeth', 'bvdu',
    'thapar university',
    'lnmiit',
    'iiit delhi',
    'iiit allahabad',
    'nit ', 'nit trichy', 'nit surathkal', 'nit warangal',
    'national institute of technology',
    'dtu', 'delhi technological university',
    'nsut', 'netaji subhas university of technology',
    'flame university',
    'jindal global university',
    'arizona state university',
    'university of texas dallas',
    'rutgers university',
    'penn state',
    'university of florida',
    'georgia tech',
    'university of manchester',
    'university of edinburgh',
    'king’s college london',
    'university of bristol',
    'university of glasgow',
    'university of waterloo',
    'mcgill university',
    'university of british columbia',
    'delft university of technology',
    'kaist',
    'seoul national university',
    'hong kong university of science and technology',
    'university of tokyo',
    'kyoto university'
]

COMPANY_MNC = [
    'google', 'microsoft', 'amazon', 'meta', 'apple', 'netflix',
    'ibm', 'oracle', 'sap', 'accenture', 'infosys', 'tcs',
    'wipro', 'cognizant', 'deloitte', 'pwc', 'kpmg', 'ey',
    'mckinsey', 'bcg', 'bain', 'jp morgan', 'goldman sachs',
    'morgan stanley', 'citibank', 'hsbc', 'bosch', 'siemens',
    'central railway', 'adobe', 'salesforce', 'uber', 'airbnb',
    'tata consultancy', 'tata consultancy services',
]

COMPANY_STARTUP = [
    'startup', 'early stage', 'seed stage', 'series a', 'series b',
    'bootstrapped', 'co-founder', 'entrepreneur',
]

TOOLS_LIST = [
    'git', 'github', 'docker', 'kubernetes', 'jenkins', 'aws',
    'azure', 'gcp', 'linux', 'figma', 'jira', 'postman',
    'tensorflow', 'pytorch', 'opencv', 'firebase', 'mongodb',
    'postgresql', 'mysql', 'redis', 'nginx', 'flask', 'django',
    'fastapi', 'nodejs', 'express', 'hibernate',
    'netlify', 'vercel', 'heroku', 'canva', 'photoshop',
    'autocad', 'tableau', 'powerbi', 'vite', 'tailwind',
    'framer', 'webpack', 'unity', 'arcore', 'android studio',
]

# National competitive exams — show high academic ability
NATIONAL_EXAMS = [
    'gate', 'ugc net', 'ugc-net', 'net ', 'upsc',
    'cat ', 'gre', 'gmat', 'toefl', 'ielts',
    'jest', 'jam ', 'csir', 'slet', 'set ',
]

SECTION_HEADERS = [
    'education', 'experience', 'professional experience',
    'work experience', 'projects', 'skills', 'achievements',
    'certifications', 'publications', 'awards', 'research',
    'research experience', 'teaching', 'courses', 'scholastic',
    'positions of responsibility', 'extra curricular',
    'extracurricular', 'leadership', 'volunteer',
]

PUBLISHER_WORDS = [
    'springer', 'elsevier', 'ieee', 'acm press',
    'singapore', 'new york', 'london', 'wiley',
]

SKIP_RESEARCH_EXACT = [
    'publications and patents', 'publications',
    'research experience', 'research', 'published work',
    'springer,', 'springer', 'singapore', 'elsevier',
    'springer singapore', 'ieee press',
]


# ----------------------------------------------------------------
# TEXT EXTRACTION
# ----------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    import io
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    import io
    doc  = docx.Document(io.BytesIO(file_bytes))
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return text.strip()


def extract_text(file_bytes: bytes, filename: str) -> str:
    fname = filename.lower()
    if fname.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif fname.endswith('.docx'):
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type: {filename}. Use .pdf or .docx")


# ----------------------------------------------------------------
# NLTK HELPER
# ----------------------------------------------------------------

def lemmatize_word(word: str, pos: str = 'n') -> str:
    pos_map = {'J': 'a', 'V': 'v', 'N': 'n', 'R': 'r'}
    wn_pos  = pos_map.get(pos[0].upper(), 'n') if pos else 'n'
    return LEMMATIZER.lemmatize(word.lower(), wn_pos)


# ----------------------------------------------------------------
# SECTION DETECTOR
# ----------------------------------------------------------------

def detect_sections(text: str) -> dict:
    lines    = text.split('\n')
    sections = {}
    current  = 'header'
    sections[current] = []

    for line in lines:
        ls       = line.strip()
        ls_lower = ls.lower()

        is_header = False
        if 2 < len(ls) < 40:
            for hdr in SECTION_HEADERS:
                if re.search(r'\b' + re.escape(hdr) + r'\b', ls_lower):
                    current   = hdr
                    is_header = True
                    if current not in sections:
                        sections[current] = []
                    break

        if not is_header:
            sections[current].append(line)

    return sections


# ----------------------------------------------------------------
# INDIVIDUAL EXTRACTORS
# ----------------------------------------------------------------

def extract_cgpa(text: str) -> float:
    """
    Extracts CGPA or percentage from the BACHELOR'S degree only.
    Priority:
    1. CGPA on Bachelor's line
    2. Percentage on Bachelor's line -> converted to CGPA (/9.5)
    3. Fallback: explicit CGPA anywhere
    4. Fallback: lowest percentage anywhere -> converted
    """
    lines      = text.split('\n')
    text_lower = text.lower()

    CGPA_PATTERNS = [
        r'cgpa\s*[:\-]?\s*([0-9]+\.?[0-9]*)\s*/\s*10',
        r'cgpa\s*[:\-]?\s*([0-9]+\.?[0-9]*)',
        r'([0-9]+\.?[0-9]*)\s*/\s*10',
        r'([0-9]+\.?[0-9]*)\s*cgpa',
        r'gpa\s*[:\-]?\s*([0-9]+\.?[0-9]*)\s*/\s*4',
        r'gpa\s*[:\-]?\s*([0-9]+\.?[0-9]*)',
    ]

    PERCENT_PATTERNS = [
        r'([0-9]{2,3}\.?[0-9]*)\s*%',
        r'([0-9]{2,3}\.?[0-9]*)\s*percent',
        r'percentage\s*[:\-]?\s*([0-9]{2,3}\.?[0-9]*)',
        r'marks\s*[:\-]?\s*([0-9]{2,3}\.?[0-9]*)',
        r'aggregate\s*[:\-]?\s*([0-9]{2,3}\.?[0-9]*)',
        r'score\s*[:\-]?\s*([0-9]{2,3}\.?[0-9]*)',
    ]

    def pct_to_cgpa(pct: float) -> float:
        return round(min(pct / 9.5, 10.0), 2)

    def try_cgpa(line: str):
        ll = line.lower()
        for pattern in CGPA_PATTERNS:
            m = re.search(pattern, ll)
            if m:
                val = float(m.group(1))
                if 'gpa' in pattern and val <= 4.0:
                    val = round(val * 2.5, 2)
                if 0.0 <= val <= 10.0:
                    return round(val, 2)
                elif 10.0 < val <= 100.0:
                    return round(val / 10, 2)
        return None

    def try_pct(line: str):
        ll = line.lower()
        for pattern in PERCENT_PATTERNS:
            m = re.search(pattern, ll)
            if m:
                val = float(m.group(1))
                if 40.0 <= val <= 100.0:
                    return val
        return None

    # Step 1: Bachelor's line
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in BACHELORS_KEYWORDS):
            cgpa = try_cgpa(line)
            if cgpa:
                return cgpa
            pct = try_pct(line)
            if pct:
                return pct_to_cgpa(pct)
            if i + 1 < len(lines):
                cgpa = try_cgpa(lines[i + 1])
                if cgpa:
                    return cgpa
                pct = try_pct(lines[i + 1])
                if pct:
                    return pct_to_cgpa(pct)

    # Step 2: Explicit CGPA anywhere
    for pattern in CGPA_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            val = float(m.group(1))
            if 'gpa' in pattern and val <= 4.0:
                val = round(val * 2.5, 2)
            if 0.0 <= val <= 10.0:
                return round(val, 2)
            elif 10.0 < val <= 100.0:
                return round(val / 10, 2)

    # Step 3: Lowest percentage anywhere
    all_pcts = []
    for pattern in PERCENT_PATTERNS:
        for m in re.finditer(pattern, text_lower):
            val = float(m.group(1))
            if 40.0 <= val <= 100.0:
                all_pcts.append(val)

    if all_pcts:
        return pct_to_cgpa(min(all_pcts))

    return 0.0


def extract_experience_years(text: str) -> float:
    """
    Calculate WORK experience only.
    Handles:
    - Standard:   'Aug 2015 - Jul 2017'
    - Year-first: '2015 Aug - 2017 July' (common in Indian resumes)
    - Split line: 'Dec 2022 - Present' across two lines
    - Till Date:  '2022 Mar-Till Date'
    """
    import datetime

    current_year  = datetime.datetime.now().year
    current_month = datetime.datetime.now().month

    MONTHS = {
        'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
        'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,
        'january':1,'february':2,'march':3,'april':4,'june':6,
        'july':7,'august':8,'september':9,'october':10,
        'november':11,'december':12,
    }

    sections  = detect_sections(text)
    WORK_SECS = ['experience', 'professional experience',
                 'work experience', 'research experience', 'research']

    work_lines = []
    for sec_name, sec_lines in sections.items():
        if any(ws in sec_name for ws in WORK_SECS):
            work_lines.extend(sec_lines)

    if not work_lines:
        lines_lower  = text.lower().split('\n')
        in_education = False
        edu_idxs     = set()
        for i, line in enumerate(lines_lower):
            ls = line.strip()
            if re.search(r'\beducation\b', ls) and len(ls) < 30:
                in_education = True
            elif any(re.search(r'\b' + h + r'\b', ls)
                     for h in ['experience','projects','skills',
                                'achievements','publications']):
                if len(ls) < 30:
                    in_education = False
            if in_education:
                edu_idxs.add(i)
        work_lines = [l for i, l in enumerate(lines_lower)
                      if i not in edu_idxs]

    work_text  = '\n'.join(work_lines)
    work_text  = re.sub(r'\(cid:[^)]+\)', ' ', work_text)
    work_text  = re.sub(r'\s+', ' ', work_text).lower()

    full_clean = re.sub(r'\(cid:[^)]+\)', ' ', text.lower())
    full_clean = re.sub(r'\s+', ' ', full_clean)

    total_months = 0
    seen         = set()

    # Pattern 1: Standard Month Year - Month/Present
    pattern_month = (
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
        r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|'
        r'dec(?:ember)?)\s+(\d{4})\s*[-–]\s*'
        r'(?:.{0,80}?\s+)?'
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
        r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|'
        r'dec(?:ember)?|present|current|now|till\s*date|to\s*date)\s*(\d{4})?'
    )

    for match in re.finditer(pattern_month, work_text):
        sm = MONTHS.get(match.group(1)[:3], 1)
        sy = int(match.group(2))
        es = match.group(3).replace(' ', '')
        if es in ('present','current','now','tilldate','todate'):
            em, ey = current_month, current_year
        else:
            em = MONTHS.get(es[:3], 1)
            ey = int(match.group(4)) if match.group(4) else current_year
        if sy < 1990 or sy > current_year: continue
        if ey < sy or ey > current_year + 1: continue
        key = (sy, sm, ey, em)
        if key in seen: continue
        seen.add(key)
        months = (ey - sy) * 12 + (em - sm)
        if 0 < months <= 120:
            total_months += months

    # Pattern 2: Year-first format "2015 Aug-2017 July"
    pattern_year_first = (
        r'(\d{4})\s+'
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
        r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|'
        r'dec(?:ember)?)\s*[-–]\s*'
        r'(\d{4})?\s*'
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
        r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|'
        r'dec(?:ember)?|present|current|now|till\s*date|to\s*date)'
    )

    for match in re.finditer(pattern_year_first, full_clean):
        sy     = int(match.group(1))
        sm     = MONTHS.get(match.group(2)[:3], 1)
        ey_str = match.group(3)
        es     = match.group(4).replace(' ', '')

        if es in ('present','current','now','tilldate','todate'):
            em, ey = current_month, current_year
        else:
            em = MONTHS.get(es[:3], 1)
            ey = int(ey_str) if ey_str else current_year

        if sy < 1990 or sy > current_year: continue
        if ey < sy or ey > current_year + 1: continue
        key = (sy, sm, ey, em)
        if key in seen: continue
        seen.add(key)
        months = (ey - sy) * 12 + (em - sm)
        if 0 < months <= 120:
            total_months += months

    # Pattern 3: Year-only fallback
    if total_months == 0:
        pattern_year = r'(\d{4})\s*[-–]\s*(present|till\s*date|\d{4})'
        for match in re.finditer(pattern_year, work_text):
            sy  = int(match.group(1))
            end = match.group(2).replace(' ', '')
            ey  = current_year if end in ('present','tilldate') else int(end)
            if sy < 2000 or ey - sy > 8: continue
            key = (sy, ey)
            if key in seen: continue
            seen.add(key)
            months = (ey - sy) * 12
            if 0 < months <= 96:
                total_months += months

    if total_months > 0:
        return round(min(total_months / 12, 20.0), 2)

    match = re.search(
        r'(\d+\.?\d*)\s*\+?\s*years?\s+of\s+experience', text.lower())
    if match:
        return float(match.group(1))

    return 0.0


def extract_internships(text: str) -> int:
    text_lower    = text.lower()
    collapsed     = re.sub(r'\s+', ' ', text_lower)
    count         = 0
    matched_spans = []

    for pattern in INTERNSHIP_ROLE_PATTERNS:
        for match in re.finditer(pattern, collapsed):
            start, end = match.span()
            overlap = any(abs(start - ms) < 50 for ms in matched_spans)
            if not overlap:
                context = collapsed[max(0, start-100):end+100]
                if re.search(r'\d{4}', context):
                    matched_spans.append(start)
                    count += 1

    return min(count, 6)


def extract_projects(text: str) -> int:
    """
    Count distinct projects.
    Only looks inside Projects section.
    Returns 0 if no projects section found.
    """
    sections      = detect_sections(text)
    project_lines = []

    for sec_name, sec_lines in sections.items():
        if 'project' in sec_name:
            project_lines = sec_lines
            break

    if not project_lines:
        return 0

    count       = 0
    seen_titles = set()

    date_prefix = re.compile(
        r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|'
        r'january|february|march|april|june|july|august|september|'
        r'october|november|december|\d{4})\b',
        re.IGNORECASE
    )

    for line in project_lines:
        ls       = line.strip()
        ls_lower = ls.lower()

        if len(ls) < 5:
            continue

        if any(skip in ls_lower for skip in
               ['tech stack', 'github :', 'http', '(cid:',
                'formulated', 'developed a', 'built a',
                'implemented', 'surveyed', 'analysed',
                'designed a', 'empirically', 'incorporated']):
            continue

        if 'indian institute' in ls_lower:
            continue

        # Format A: ends with ':'
        if ls.endswith(':') and 5 < len(ls) < 100:
            title = ls.rstrip(':').strip().lower()
            if title not in seen_titles:
                seen_titles.add(title)
                count += 1
            continue

        # Format B: date-prefixed
        date_match = date_prefix.match(ls)
        if date_match:
            remainder = ls[date_match.end():].strip()
            remainder = re.sub(r'^[-–\d\s,]+', '', remainder).strip()
            if 5 < len(remainder) < 80:
                title = remainder.lower()
                if title not in seen_titles:
                    seen_titles.add(title)
                    count += 1
            continue

        # Format C: bullet point
        if ls.startswith(('•', '-', '*', '◦', '▪')):
            remainder = ls[1:].strip()
            if 5 < len(remainder) < 80 and not any(
                    kw in remainder.lower() for kw in
                    ['developed', 'built', 'implemented', 'designed',
                     'formulated', 'using', 'based on']):
                title = remainder.lower()
                if title not in seen_titles:
                    seen_titles.add(title)
                    count += 1

    return min(max(count, 0), 15)


def extract_programming_languages(text: str) -> int:
    text_lower = text.lower()
    found = set()
    for lang in PROGRAMMING_LANGUAGES:
        if len(lang) <= 2:
            if re.search(
                    r'(?<![a-z])' + re.escape(lang) + r'(?![a-z])',
                    text_lower):
                found.add(lang)
        else:
            if re.search(r'\b' + re.escape(lang) + r'\b', text_lower):
                found.add(lang)
    return min(len(found), 10)


def extract_certifications(text: str) -> int:
    """
    NLTK-enhanced: lemmatization + skip award/hackathon contexts.
    """
    text_lower = text.lower()
    count      = 0
    seen       = set()

    SKIP_CONTEXTS = [
        'hackathon', 'runner-up', 'runner up', 'contest',
        'competition', 'first place', 'second place', 'winner',
        'shaastra', 'techfest', 'achievement', 'award',
        'rank', 'prize', 'state rank',
    ]

    for line in text_lower.split('\n'):
        line = line.strip()
        if line in seen or len(line) < 5:
            continue

        if any(skip in line for skip in SKIP_CONTEXTS):
            continue

        if any(kw in line for kw in CERTIFICATION_KEYWORDS):
            count += 1
            seen.add(line)
            continue

        try:
            tokens = word_tokenize(line)
            tagged = pos_tag(tokens)
            for word, pos in tagged:
                lemma = lemmatize_word(word, pos)
                if lemma in CERTIFICATION_LEMMAS and line not in seen:
                    if not any(skip in line for skip in SKIP_CONTEXTS):
                        count += 1
                        seen.add(line)
                        break
        except Exception:
            pass

    return min(count, 5)


def extract_hackathons(text: str) -> int:
    """Count distinct hackathon events."""
    text_lower  = text.lower()
    event_lines = set()

    for line in text_lower.split('\n'):
        line = line.strip()
        if not line:
            continue
        for kw in HACKATHON_KEYWORDS:
            if kw in line:
                event_lines.add(line[:60])
                break

    return min(len(event_lines), 5)


def extract_research_papers(text: str) -> int:
    """
    Count distinct research publications.
    Skips section headers, publisher-only lines.
    """
    text_lower = text.lower()
    count      = 0
    seen_lines = set()

    for line in text_lower.split('\n'):
        line = line.strip()

        if not line or line in seen_lines:
            continue

        if line in SKIP_RESEARCH_EXACT:
            continue

        if len(line) < 20 and any(pw in line for pw in PUBLISHER_WORDS):
            continue

        if len(line) < 25 and any(s in line for s in SKIP_RESEARCH_EXACT):
            continue

        for kw in RESEARCH_KEYWORDS:
            if kw in line:
                count += 1
                seen_lines.add(line)
                break

    return min(count, 5)


def extract_soft_skills_score(text: str) -> float:
    """
    NLTK-enhanced: direct keyword matching + POS tagging + lemmatization.
    """
    text_lower = text.lower()

    found_direct = set()
    for kw in SOFT_SKILL_KEYWORDS:
        if kw in text_lower:
            found_direct.add(kw)

    found_nltk = set()
    try:
        sentences = sent_tokenize(text)
        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            tagged = pos_tag(tokens)
            for word, pos in tagged:
                if pos in SOFT_SKILL_POS:
                    lemma = lemmatize_word(word, pos)
                    if lemma in SOFT_SKILL_LEMMAS:
                        found_nltk.add(lemma)
    except Exception:
        pass

    total_found = len(found_direct) + len(found_nltk - {
        LEMMATIZER.lemmatize(w.split()[0])
        for w in found_direct
    })

    base  = 3.0 if 'skill' in text_lower else 0.0
    score = base + total_found * 0.5
    return round(min(score, 10.0), 2)


def extract_education(text: str) -> tuple:
    text_lower  = text.lower() + ' '
    edu_enc     = 1

    for level, keywords in EDUCATION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            if level == 'PhD':
                edu_enc = 3
                break
            elif level == 'Masters':
                edu_enc = 2

    tier_enc    = 1
    tier1_found = False

    for kw in TIER1_UNIVERSITIES:
        kw_clean = kw.strip()
        if len(kw_clean) <= 4:
            if re.search(r'\b' + re.escape(kw_clean) + r'\b', text_lower):
                tier1_found = True
                break
        else:
            if kw in text_lower:
                tier1_found = True
                break

    if tier1_found:
        tier_enc = 3
    else:
        for kw in TIER2_UNIVERSITIES:
            if kw in text_lower:
                tier_enc = 2
                break

    return edu_enc, tier_enc


def extract_company_type(text: str) -> int:
    text_lower = text.lower()
    if any(kw in text_lower for kw in COMPANY_MNC):
        return 3
    elif any(kw in text_lower for kw in COMPANY_STARTUP):
        return 1
    return 2


def extract_age(text: str) -> int:
    import datetime
    patterns = [
        r'age\s*[:\-]?\s*(\d{2})',
        r'(\d{2})\s*years?\s+old',
        r'dob\s*[:\-]?\s*\d{1,2}[\/\-]\d{1,2}[\/\-](\d{4})',
        r'date\s+of\s+birth\s*[:\-]?\s*\d{1,2}[\/\-]\d{1,2}[\/\-](\d{4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            val = int(match.group(1))
            if val > 1900:
                return datetime.datetime.now().year - val
            if 18 <= val <= 60:
                return val
    return 22


def extract_skills_score(text: str) -> float:
    """
    NLTK-enhanced skills score including academic achievements.
    Handles both technical profiles AND academic/research profiles.

    Formula:
      prog_languages  x 1.5   (technical languages)
      tool_count      x 0.5   (tools and frameworks)
      certifications  x 0.8   (professional certs/FDPs)
      hackathons      x 0.5   (competitions)
      national_exams  x 1.2   (GATE/UGC NET/UPSC etc.)
      research_bonus          (publications/thesis)

    This ensures academic candidates (with GATE, UGC NET, publications)
    score fairly alongside software engineers (with tools and languages).
    """
    text_lower = text.lower()

    # NLTK tokenize + lemmatize for tool matching
    try:
        tokens          = word_tokenize(text_lower)
        tokens_filtered = [t for t in tokens
                           if t not in STOP_WORDS and t.isalpha()]
        lemmas          = set(LEMMATIZER.lemmatize(t)
                              for t in tokens_filtered)
    except Exception:
        lemmas = set(text_lower.split())

    prog     = min(extract_programming_languages(text), 5)
    cert     = extract_certifications(text)
    hack     = extract_hackathons(text)
    research = extract_research_papers(text)

    # Count tools using lemmatized tokens
    tool_count = 0
    for t in TOOLS_LIST:
        t_lemma = LEMMATIZER.lemmatize(t.lower())
        if t_lemma in lemmas or t in text_lower:
            tool_count += 1
    tool_count = min(tool_count, 12)

    # Count national competitive exams
    exam_count = 0
    for ex in NATIONAL_EXAMS:
        if ex in text_lower:
            exam_count += 1
    exam_count = min(exam_count, 3)

    # Research/publication bonus (each paper = 0.8, max 3.0)
    research_bonus = min(research * 0.8, 3.0)

    score = (
        prog           * 1.5 +
        tool_count     * 0.5 +
        cert           * 0.8 +
        hack           * 0.5 +
        exam_count     * 1.2 +
        research_bonus
    )

    return round(min(max(score, 2.0), 38.5), 1)


def extract_resume_word_count(text: str) -> int:
    """
    NLTK-enhanced: tokenization + stopword removal.
    """
    try:
        tokens   = word_tokenize(text.lower())
        filtered = [t for t in tokens
                    if t.isalpha() and t not in STOP_WORDS]
        return max(0, len(filtered))
    except Exception:
        return max(0, len(text.split()))


# ----------------------------------------------------------------
# MASTER PARSER
# ----------------------------------------------------------------

def parse_resume(file_bytes: bytes, filename: str) -> dict:
    text = extract_text(file_bytes, filename)

    if not text or len(text.strip()) < 50:
        raise ValueError(
            "Could not extract meaningful text from the resume file.")

    edu_enc, tier_enc = extract_education(text)

    features = {
        'age'                  : extract_age(text),
        'cgpa'                 : extract_cgpa(text),
        'internships'          : extract_internships(text),
        'projects'             : extract_projects(text),
        'programming_languages': extract_programming_languages(text),
        'certifications'       : extract_certifications(text),
        'experience_years'     : extract_experience_years(text),
        'hackathons'           : extract_hackathons(text),
        'research_papers'      : extract_research_papers(text),
        'skills_score'         : extract_skills_score(text),
        'soft_skills_score'    : extract_soft_skills_score(text),
        'resume_length_words'  : extract_resume_word_count(text),
        'edu_enc'              : edu_enc,
        'tier_enc'             : tier_enc,
        'comp_enc'             : extract_company_type(text),
    }

    return {
        'extracted_features': features,
        'raw_text_preview'  : text[:500] + '...' if len(text) > 500 else text,
        'word_count'        : features['resume_length_words'],
    }