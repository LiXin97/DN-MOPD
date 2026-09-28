# SPDX-License-Identifier: Apache-2.0
"""Evaluation protocol of the DN-MOPD paper: generation (vLLM), official graders, aggregation and paired bootstrap.

Command-line entry points:
  python -m eval.generate   sample answers for one model, suite and cap
  python -m eval.grade      grade a generation directory with the official checkers
  python -m eval.aggregate  suite, domain and Total scores, mean output tokens and cap-hit rate
  python -m eval.compare    paired bootstrap difference between two models
"""
