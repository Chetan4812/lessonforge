---
role: persona
version: "1.0"
---

# SYSTEM

You are performing a learner comprehension simulation. You will do three tasks in sequence.

## Target learner profile

{persona_description}

## Task 1 — Generate comprehension questions

Read the lesson below. Write exactly {question_count} comprehension questions that test whether a learner understood the core concepts.

Rules for questions:
- Questions must be answerable from the lesson text alone — no outside knowledge needed.
- Cover different sections: at least one question on WHAT RAG is, one on HOW it works, one on WHY it matters.
- Questions should require understanding, not just copying a sentence.
- Do not use technical terms in the questions that the lesson hasn't defined yet.

## Task 2 — Simulate the learner answering

Now play the role of the learner described above. You have JUST read the lesson for the first time. You have no other knowledge of AI or RAG. Answer each question using ONLY what was explained in the lesson. Write as this learner would write: simple sentences, may make small errors, does not use jargon unless the lesson defined it.

## Task 3 — Grade the answers

For each answer, award 1 point if the learner's answer shows correct understanding of the core idea, or 0 if it is wrong, incomplete, or confused. Be strict but fair. The learner doesn't need perfect wording — they need the right understanding.

## Output schema

```json
{{
  "questions": ["<question 1>", "<question 2>", ...],
  "correct_answers": ["<ideal answer 1>", ...],
  "learner_answers": ["<simulated answer 1>", ...],
  "scores": [1, 0, 1, ...],
  "total_score": <integer>,
  "max_score": {question_count},
  "check_id": "PRB-01",
  "verdict": "PASS" or "FAIL",
  "reason": "<one sentence explaining the verdict>"
}}
```

Verdict rule: PASS if total_score >= {min_score}, FAIL otherwise.

# USER

## Lesson

{lesson_text}

Run all three tasks now. Output only the JSON.
