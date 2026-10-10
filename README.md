# 2-loop-Corrections-to-Statistical-Prefactor
Numerical computation of the two-loop statistical (fluctuation) contribution to the false vacuum decay rate, using Green's function methods on a bounce/instanton background.

**Status**: work in progress. The write-up is in preparation. Results, code structure and interfaces may change. Last updated: <!-- DATE --> 10/10/26

## Overview

This project extends the semi-classical treatment of metastable decay beyond one loop. The two-loop contribution to the escape rate is built from three diagrams:

- double bubble
- dumbbell
- sunset

Main ingredients:

- Partial-wave decomposition of the real-space propagators, with ℓ = 0, ℓ = 1 and ℓ ≥ 2 treated separately.
- Reduced Green's function for the ℓ = 1 sector, accounting for the translational zero mode (vanishing Wronskian).
- Numerical evaluation using Riccati shooting.
- Renormalisation of the two-loop terms via counterterms (MS-bar and dim-reg).

A preprint will be linked here when available.

Method outline
- Construct the bounce profile.
- Compute the Green's function for each partial wave in the bounce and metastable backgrounds.
- Expansion of the bounce propagator in terms of the metastable propagator, leading to different objects, including the WKB expansion difference between the two propagators, and the difference between the propagator solution difference and the WKB approximation difference.
- Sum over partial waves and evaluate each diagram.
- Combine with counterterms to obtain the renormalised two-loop correction.


>**_ Repository structure
.
├── 2-loop_corrections_2_0_3.ipynb   # current main notebook (all three diagrams)
├── archive/                         # older notebook versions
├── figures/                         # output plots
├── requirements.txt                 # Python dependencies
└── README.md
<!-- Edit this tree to match what is actually in the repo. -->
Requirements
Python <!-- version -->
Packages: NumPy, SciPy, mpmath, joblib, Jupyter <!-- check against requirements.txt -->
Memory note: the sunset and dumbbell calculations use large 3D arrays and can run out of memory on small machines. <!-- add hardware used / BLAS thread settings -->

Install with:

bash
pip install -r requirements.txt
Usage

Open the main notebook and run all cells for a chosen coupling value (lambdap):

bash
jupyter notebook 2-loop_corrections_2_0_3.ipynb
<!-- Once the code is packaged, replace this with a short example, e.g. from twoloop import run_pipeline result = run_pipeline(lambdap=0.1) -->
Progress
 Double bubble
 Dumbbell
 Sunset
 Convergence of the partial-wave sums
 MS-bar renormalisation of the two-loop terms
 Validation against known results
 Package the pipeline as an importable module
 Final scans and figures
Known issues
Convergence of some partial-wave sums is still being investigated.
References
B. Garbrecht and P. Millington, arXiv:1501.07466, arXiv:1804.04944
M. Carosi and B. Garbrecht, arXiv:2411.18421, arXiv:2601.08686
A. Andreassen et al., arXiv:1604.06090
O. Gould, BubbleDet, arXiv:2308.15652
Citation

Preprint in preparation.

Author

David, University of Nottingham. <!-- add contact or GitHub handle -->

License
<!-- Choose a licence (e.g. MIT) after checking with your supervisors. --> _**>
