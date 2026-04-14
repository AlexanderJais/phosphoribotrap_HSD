"""Phosphoribotrap RNA-seq analysis package.

A Streamlit-backed pipeline for phosphorylated-ribosome IP RNA-seq in a
3-group matched design (NCD / HSD1 / HSD3, 3 biological replicates each,
each animal contributing one IP and one INPUT library).

Primary differential-translation analysis is anota2seq (shelled out via
Rscript); a Python-side sign-consistency + between-group Mann-Whitney
cross-check is always available without the R stack.
"""

__version__ = "0.1.0"
