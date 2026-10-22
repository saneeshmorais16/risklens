# RiskLens

Synthetic transaction review console. An Isolation Forest is fit on the early days of a generated book, frozen, and used to score the holdout. Reviewers can clear, hold, or escalate a flagged row, watch drift and slice gaps, and ask a voice assistant about the open case.

A high score is a reason to look, not a finding of fraud. The book is synthetic.

## Run

```bash
npm install
npm run dev
```

The app listens on port 8080. Voice answers need `XAI_API_KEY` in the environment. Do not commit that key.
