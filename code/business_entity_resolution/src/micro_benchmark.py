"""
Micro-benchmark for fast candidate ranking function.
"""
import time
import re
from rapidfuzz import fuzz

s1_name = "Orelee's Barbershop"
s1_addr = "1795 Westchester Drive, High Point, NC"
s1_cntry = "US"

noisy_name = "Orelee'S Services"
noisy_addr = "Westchester Dr, High Point, North Carolina"
noisy_cntry = "US"

RE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
RE_DIGITS = re.compile(r"\b\d+\b")

p_s1_n = RE_PUNCT.sub(" ", s1_name.lower()).strip()
p_m_n = RE_PUNCT.sub(" ", noisy_name.lower()).strip()
p_s1_a = RE_PUNCT.sub(" ", s1_addr.lower()).strip()
p_m_a = RE_PUNCT.sub(" ", noisy_addr.lower()).strip()

s1_n_toks = set(p_s1_n.split())
m_n_toks = set(p_m_n.split())
s1_a_toks = set(p_s1_a.split())
m_a_toks = set(p_m_a.split())

n_jacc = len(s1_n_toks & m_n_toks) / len(s1_n_toks | m_n_toks) if (s1_n_toks and m_n_toks) else 0.0
a_jacc = len(s1_a_toks & m_a_toks) / len(s1_a_toks | m_a_toks) if (s1_a_toks and m_a_toks) else 0.0

n_fuzz = fuzz.ratio(p_s1_n, p_m_n) / 100.0
a_fuzz = fuzz.ratio(p_s1_a, p_m_a) / 100.0

s1_nums = set(RE_DIGITS.findall(p_s1_a))
m_nums = set(RE_DIGITS.findall(p_m_a))
num_match = 1.0 if (s1_nums and m_nums and bool(s1_nums & m_nums)) else 0.0

cntry_match = 1.0 if s1_cntry.upper() == noisy_cntry.upper() else 0.0

ranking_score = (
    0.40 * n_fuzz
  + 0.25 * a_fuzz
  + 0.15 * n_jacc
  + 0.10 * a_jacc
  + 0.05 * num_match
  + 0.05 * cntry_match
)

print(f"Ranking score: {ranking_score:.4f} (Name Fuzz: {n_fuzz:.2f}, Addr Fuzz: {a_fuzz:.2f}, N Jacc: {n_jacc:.2f}, A Jacc: {a_jacc:.2f})")

# Test 100,000 iterations speed
t0 = time.time()
for _ in range(100_000):
    nf = fuzz.ratio(p_s1_n, p_m_n)
    af = fuzz.ratio(p_s1_a, p_m_a)
    sc = 0.40 * nf + 0.30 * af
elapsed = time.time() - t0
print(f"100,000 fuzz scoring operations took: {elapsed:.3f}s ({100_000/elapsed:,.0f} pairs/sec)")
