"""Human-to-mouse conservation map for TfR1, used to score where designs land.

Criterion 2 ranks designs by how close the mouse/human KD ratio is to 1, so the
mouse-conserved fraction of whatever surface a design actually binds is the
thing to measure. Overall identity is 77.7%, but it is distributed unevenly:
the recommended apical epitope is 2 of 9 conserved while a patch centred on
residue 322, also in the apical domain, is 88% conserved.

Alignment is Needleman-Wunsch with identity scoring, which is adequate at 78%
identity. Cached to conservation.json so the design and scoring paths do not
re-derive it.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
HUMAN_FASTA = HERE / "sequences" / "P02786.fasta"
MOUSE_FASTA = HERE / "sequences" / "Q62351.fasta"
CACHE = HERE / "conservation.json"


def _read(p):
    return "".join(l.strip() for l in open(p) if not l.startswith(">"))


def align(H, M, gap=-8, match=5, mismatch=-4):
    n, m = len(H), len(M)
    S = np.zeros((n + 1, m + 1), dtype=np.int32)
    S[0, :] = np.arange(m + 1) * gap
    S[1:, 0] = np.arange(1, n + 1) * gap
    P = np.zeros((n + 1, m + 1), dtype=np.int8)
    MB = np.frombuffer(M.encode(), dtype="S1")
    for i in range(1, n + 1):
        row = S[i - 1, :-1] + np.where(MB == H[i - 1].encode(), match, mismatch)
        for j in range(1, m + 1):
            d, u, l = row[j - 1], S[i - 1, j] + gap, S[i, j - 1] + gap
            if d >= u and d >= l:
                S[i, j], P[i, j] = d, 0
            elif u >= l:
                S[i, j], P[i, j] = u, 1
            else:
                S[i, j], P[i, j] = l, 2
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        if P[i, j] == 0:
            pairs.append((i, j)); i -= 1; j -= 1
        elif P[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return dict(pairs)


def conservation(rebuild: bool = False) -> dict:
    """{human_resnum: {'human','mouse','same','mouse_num'}} over the alignment."""
    if CACHE.exists() and not rebuild:
        return {int(k): v for k, v in json.loads(CACHE.read_text())["map"].items()}
    H, M = _read(HUMAN_FASTA), _read(MOUSE_FASTA)
    h2m = align(H, M)
    out = {}
    for p in range(1, len(H) + 1):
        mj = h2m.get(p)
        out[p] = {"human": H[p - 1],
                  "mouse": (M[mj - 1] if mj else None),
                  "mouse_num": mj,
                  "same": bool(mj and H[p - 1] == M[mj - 1])}
    ident = sum(1 for v in out.values() if v["same"]) / sum(
        1 for v in out.values() if v["mouse"])
    CACHE.write_text(json.dumps({"identity_over_aligned": ident,
                                 "map": {str(k): v for k, v in out.items()}}))
    return out


def conserved_fraction(resnums) -> float:
    """Fraction of `resnums` identical in mouse. Indels count as divergent."""
    c = conservation()
    rs = [r for r in resnums if r in c]
    return (sum(1 for r in rs if c[r]["same"]) / len(rs)) if rs else 0.0


if __name__ == "__main__":
    c = conservation(rebuild=True)
    tot = sum(1 for v in c.values() if v["mouse"])
    print(f"aligned {tot} positions, identity "
          f"{sum(1 for v in c.values() if v['same'])/tot:.1%}")
    APICAL = [202, 204, 208, 209, 210, 211, 212, 348, 371]
    print(f"apical epitope (11rn chain X): {conserved_fraction(APICAL):.0%} conserved")
    print(f"patch 317-325:                 {conserved_fraction(range(317,326)):.0%} conserved")
