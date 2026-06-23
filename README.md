# Machine Learning Toolbox — Python Companion

A chapter-by-chapter **Python implementation** of the code from
**Yigit Aydede, _Machine Learning Toolbox for Social Scientists_** (Taylor & Francis, 2023).

It is part of the open-textbook project *Introduction to Applied Machine Learning with
R and Python for Social Sciences and Business*, developed with support from
**[AtlanticOER](https://atlanticoer-reloa.ca/)**. Each script mirrors a chapter of the
original R material, re-implemented in Python so every method is available in both
languages — giving students a single, consistent path through the same techniques in
whichever language fits their course or career.

> **Audience:** upper-year social-science, business, and graduate management students.
> The emphasis is on *applied* understanding — runnable, end-to-end examples rather than
> library-specific syntax.

---

## What's inside

All scripts live in [`python_files/`](python_files). They are meant to be read and run
alongside the corresponding chapter of the R toolbox.

| File | Topic |
|------|-------|
| `preliminaries.py` | Environment setup & data preliminaries |
| `03-Bias-VarianceTradeoff.py` | The bias–variance trade-off |
| `04-Overfitting.py` | Overfitting & model complexity |
| `05-ParametricEstimations.py` | Parametric estimation |
| `06-Basics.py` | Modelling basics |
| `07-Smoothing.py` | Smoothing methods |
| `08-Nonparametric_kNN.py` | Nonparametric estimation & k-NN |
| `09-HyperTuning.py` | Hyperparameter tuning |
| `10-TuningClass.py` | Tuning for classification |
| `11-ClassificationExample.py` | Worked classification example |
| `12-CART.py` | Classification & regression trees |
| `13-Ensemble.py` · `14-EnsembleApplication.py` | Ensemble methods & application |
| `15-SVM.py` | Support vector machines |
| `16-NN.py` | Neural networks |
| `17-Ridge.py` · `18-Lasso.py` · `19-AdaptiveL.py` · `20-Sparsity.py` | Regularization: ridge, lasso, adaptive lasso, sparsity |
| `21-TimeSeriesArima.py` · `22-TSGrid.py` · `23-TSEmbedding.py` · `24-TSRandomForest.py` | Time-series machine learning |

## Getting started

```bash
git clone https://github.com/yaydede/MLToolbox-Python.git
cd MLToolbox-Python/python_files
python preliminaries.py
```

Typical dependencies: `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `matplotlib`.

```bash
pip install numpy pandas scikit-learn statsmodels matplotlib
```

## Classroom use

The Python companion has been used in **MBAN 5560 — Machine Learning & AI** at
Saint Mary's University over the past two years, and is being introduced in
**ECON 3303 (Statistics)** and **ECON 4404 (Econometrics)**.

## Related

- *Machine Learning Toolbox for Social Scientists* — Yigit Aydede (Taylor & Francis, 2023)
- R & Python Bootcamp — https://yaydede.github.io/Bootcamp_book/

---

## Credits & acknowledgements

- **Python translation:** **Kubilay Tosuner** — translated and tested all 23 chapters.
- **Original text & R code:** **Yigit Aydede**, Saint Mary's University.
- **Project support:** developed as an open educational resource with funding from
  **[AtlanticOER](https://atlanticoer-reloa.ca/)**, whose Development Grant made the
  Python companion possible.

This work is gratefully indebted to Kubilay Tosuner's careful translation work and to
AtlanticOER's support for open, accessible learning materials.

## License

Released under **[Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)**.
The underlying text is derived from a Taylor & Francis publication; any reuse must
preserve attribution and the non-commercial restriction.
