"""List weight-host optimizer updates that do not share the common update flag.

    python cerebras/wgt_update_flags.py <ws_rt.mlir>

ws_rt.mlir: from a compile's cs_<hash>/ dir (bash cerebras/export_logs.sh <wsjob>)
or a weight host's compiles/ dir. Each wgth.adam takes an i1 flag as its last
operand; in a healthy program every parameter shares one flag per branch. The
minimal-pair penalty gave the level-0 codebook its own flag, which the weight host
read unset at iteration 0 ("Attempting to read a null buffer"; HANDOFF_MINPAIRS.md).
"""
import re
import sys
from collections import Counter

txt = open(sys.argv[1]).read().splitlines()
defs, locs = {}, {}
for line in txt:
    m = re.match(r'\s*(%\d+) = ws_rt\.wgth\.\w+.*loc\(#loc(\d+)\)\s*$', line)
    if m:
        defs[m.group(1)] = m.group(2)
    m = re.match(r'#loc(\d+) = loc\((.*)', line)
    if m:
        locs[m.group(1)] = m.group(2)
rows = [re.findall(r'%\d+', line.split('{')[0]) for line in txt if 'wgth.adam' in line]
flags = Counter(ops[-1] for ops in rows)
print("adam updates per flag:", dict(flags))
for ops in rows:
    if flags[ops[-1]] <= 2:                       # not the shared flag
        src = re.search(r'"([^"]+)"', locs.get(defs.get(ops[3], ''), ''))
        print(f"own flag {ops[-1]}: gradient from {src.group(1) if src else '?'}")
