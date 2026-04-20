# IsoSci-150 Annotation Instructions

Thank you for helping validate the IsoSci-150 benchmark.
Please read these instructions carefully before starting.

---

## What you are evaluating

Each item is an **isomorphic problem pair**: two science problems from different
domains that are claimed to share identical logical structure but require different
domain knowledge. Your job is to judge whether this claim holds.

**Example of a valid pair:**
- Source (Physics): "A 2.0 mol sample of ideal gas at 300 K occupies 49.2 L.
  What is the pressure?" (requires PV=nRT)
- Target (Chemistry): "A weak acid solution has [HA]=0.10 M and Ka=1.8×10⁻⁵.
  What is the pH?" (requires pH = -log√(Ka·C))

Both require: recall formula → substitute values → compute. But different formulas.
This is a **valid pair**.

---

## Scoring criteria (1–5 scale)

### 1. Logical Equivalence
Are the reasoning procedures identical?

| Score | Meaning |
|-------|---------|
| 5 | Identical steps: same number, same types (recall→substitute→compute) |
| 4 | Nearly identical, minor differences in complexity |
| 3 | Similar structure but one problem has extra/different steps |
| 2 | Mostly different procedures |
| 1 | Completely different reasoning required |

### 2. Domain Independence
Does knowing one problem's solution help you solve the other?

| Score | Meaning |
|-------|---------|
| 5 | Completely independent — different formulas, facts, constants |
| 4 | Mostly independent — minor conceptual adjacency |
| 3 | Some overlap — partial transfer would help |
| 2 | Significant overlap |
| 1 | Nearly identical knowledge required |

**Watch for:** same formula in different notation (e.g., ideal gas law ≈ van der Waals
at low pressure), closely related conservation laws across domains.

### 3. Difficulty Parity
Would a strong undergraduate find them equally challenging?

| Score | Meaning |
|-------|---------|
| 5 | Equal difficulty |
| 4 | Minor difference |
| 3 | One noticeably harder (more complex formula, worse numeric conditioning) |
| 2 | Moderately unequal |
| 1 | Very unequal (one trivial, one hard) |

### 4. Self-Containment
Can each problem be solved with only the given information + standard undergrad knowledge?

| Score | Meaning |
|-------|---------|
| 5 | Fully self-contained |
| 4 | Mostly self-contained |
| 3 | Minor assumption required |
| 2 | One problem missing key information |
| 1 | Neither problem is self-contained |

---

## Overall verdict

- **ACCEPT**: All four scores are ≥ 4. The pair is a valid isomorphic pair.
- **MARGINAL**: Any score is exactly 3. The pair has a minor flaw but could be included
  with caveats.
- **REJECT**: Any score is ≤ 2. The pair fails the isomorphism criterion.

---

## Failure modes

If you select MARGINAL or REJECT, please check the applicable failure modes:

- **Formula overlap**: The two formulas are too similar (e.g., related conservation laws)
- **Difficulty mismatch**: One problem is significantly harder
- **Ambiguous answer**: The correct answer is unclear or there are multiple valid answers
- **Missing information**: A problem cannot be solved with the given data
- **Trivial target**: The target problem is too simple — doesn't genuinely require recall
- **Structural inequivalence**: The solution procedure is not actually the same
- **Other**: Anything else — please explain in the notes field

---

## Practical notes

- Aim for ~5–8 minutes per pair. If you are unsure, write a note and move on.
- You do not need domain expertise in both fields — judge the **structure**, not whether
  you could solve the problem yourself.
- Score independently; do not look at the other annotator's scores until you are done.
- For the answers provided: assume they are correct. You are evaluating the pair
  structure, not checking the math.

---

## Submitting your annotations

When complete, please return:
1. The filled-in CSV file (easier for analysis), **or**
2. A screenshot/printout of the HTML booklet with scores filled in

Thank you!
