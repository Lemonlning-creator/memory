# Sources and attribution

- Deep Empathy method and original application: https://github.com/Lemonlning-creator/memory
- Experiment implementation snapshot: commit `5927bbff03fda74eebaeb99e0c57203a644cfd74` of the memory experiment history. Exact included file hashes are in `provenance/canonical_source.json`.
- REALTALK paper: Lee et al., *REALTALK: A 21-Day Real-World Dataset for Long-Term Conversation*, arXiv:2502.13270. https://arxiv.org/abs/2502.13270
- REALTALK data and public evaluation references: https://github.com/danny911kr/REALTALK at `b903e06a9770bf4e5fe9018c3e132889666d3b4a`.
- Classifier checkpoints: Cardiff NLP model IDs and pinned revisions are in `src/experiments/realtalk_evaluator.py`.
- Semantic metric: BERTScore, using `bert-score` and the English `roberta-large` configuration documented in `docs/RESULTS_ZH.md`.

This archive does not assign a new permissive license to upstream code, data, conversations, or model weights. Respect the original sources' terms and participant-data handling requirements. Do not treat model-inferred profiles as verified biographical or psychological facts. Raw conversations and inference artifacts are not included in this public Git branch; the separate results package is for the user's research handoff.

The REALTALK Persona Simulation generation/evaluation runner is a protocol-aligned implementation by this project, not an official released end-to-end runner. Judge prompt templates are retained from the frozen implementation based on Appendix C. No claim of exact author-runtime reproduction is made.
