# DATA_DICTIONARY.md — sample_colleges.csv

Reproduced from the assignment task sheet. The dataset is a synthetic set of
15 colleges (C001–C015); values are invented for the exercise and treated as
real and verified.

| Column | Type | Meaning |
| --- | --- | --- |
| college_id | string | Stable identifier, C001–C015. Used in citations. |
| name | string | Full college name. Two entries have deliberately similar names. |
| city / state | string | Location. |
| type | enum | Government \| Private \| Deemed |
| courses_offered | string | Semicolon-separated. A Diploma is not a degree. |
| annual_fees_inr | integer | Tuition in ₹ per academic year. Not per semester. Excludes hostel, mess and any charges described in "about". |
| last_year_cutoff_pct | integer | Minimum aggregate percentage admitted last year. A hard floor — a student below this figure was not eligible. |
| total_seats | integer | Total intake across all courses. |
| hostel_available | boolean | Yes \| No. |
| naac_grade | string | A++, A+, A, B++, B+, B. |
| avg_placement_lpa | float | Average package in lakhs per annum. 0 means not reported / not applicable. 0 does NOT mean "worst placements". |
| established_year | integer | Year founded. |
| about | free text | ~110 words per college: admission process, scholarships, hostel arrangements, extra charges, placement context, faculty. Unstructured. Some questions can only be answered from this field, and it sometimes qualifies or explains a structured value. |

How the pipeline honours each rule:

- **college_id** — every factual claim is cited by id; ids are verified against the retrieved context in code, and scrubbed from user-facing prose.
- **Similar names** — course exact-match + name-backing checks keep the two look-alike institutions apart (a college can only be cited for a course it actually lists).
- **A Diploma is not a degree** — diploma-only institutions may appear in relevant answers, but only with an explicit diplomas-not-degrees disclosure.
- **annual_fees_inr** — treated as per-year tuition; per-semester budgets are converted (×2) with the assumption stated; extra charges from "about" are surfaced in budget answers.
- **last_year_cutoff_pct** — enforced in code as a hard floor; eligibility verdicts are computed in code and handed to the model as facts.
- **avg_placement_lpa = 0** — rendered as "not reported / not applicable" so the model can never rank it as worst.
- **about** — mined into explicit "Support & eligibility" tags (income/gender/region-based concessions, entrance-exam waivers, etc.) so free-text-only questions retrieve reliably.
