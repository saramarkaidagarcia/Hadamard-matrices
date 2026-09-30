# Hadamard Matrix Construction via Memetic SA-GA Search

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

A high-performance GPU-accelerated search for Hadamard matrices using a hybrid **Simulated Annealing + Genetic Algorithm** metaheuristic.

## Overview

This repository implements research from the degree dissertation *"Hadamard Matrices"* (UPV/EHU, 2026). It provides a sophisticated computational toolkit for constructing Hadamard matrices of arbitrary order, with particular focus on the open **Hadamard Conjecture** (smallest unsolved order: 668).

### What are Hadamard Matrices?

A Hadamard matrix **H** of order *n* is an *n×n* matrix with entries in {+1, −1} such that:

$$HH^T = nI$$

**Properties:**
- All rows are mutually orthogonal
- Determinant maximization: |det(H)| is maximal among all {±1}-matrices of the same size
- Existence only when *n* ∈ {1, 2} or *n* ≡ 0 (mod 4)
- **Hadamard Conjecture**: A Hadamard matrix exists for every *n* ≡ 0 (mod 4)

### Applications

1. **Combinatorial Design** — Block designs, Fano plane construction
2. **Error-Correcting Codes** — Reed–Muller codes (Mars Mariner 9 spacecraft, 1971)
3. **Weighing Designs** — Variance minimization in experimental design
4. **Signal Processing** — Fast Hadamard Transform
5. **Quantum Computing** — Hadamard gates

## Algorithm

### Hybrid Memetic Approach

The search combines two powerful metaheuristics:

**1. Simulated Annealing (SA)**
- Single-bit flip moves on ±1 matrix entries
- Incremental Gram matrix updates: O(n) instead of O(n³)
- Metropolis acceptance criterion
- Temperature cooling schedule

**2. Genetic Algorithm (GA)**
- Periodic population recombination (every K SA steps)
- **Column-split crossover**: offspring inherits columns from each parent
- Bit-flip mutation on selected offspring
- Elitism: best individuals always survive

**Fitness Function:**
$$f(H) = \sum_{i \neq j} G_{ij}^2$$

where *G = HH^T*. Optimal fitness is **0** when *H* is Hadamard.

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+ (with CUDA support recommended)
- NumPy, SciPy

### Setup

```bash
# Clone the repository
git clone https://github.com/saramarkaida/hadamard-matrices.git
cd hadamard-matrices

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from hadamard_search import HadamardSearcher, HadamardSearchConfig

# Configure search for order 668
config = HadamardSearchConfig(
    matrix_order=668,
    population_size=96,
    max_restarts=60,
    device='cuda',  # or 'cpu'
)

# Run search
searcher = HadamardSearcher(config)
best_matrix, best_fitness = searcher.search()

# Results
print(f"Best fitness: {best_fitness:.0f}")
print(f"Valid Hadamard: {searcher.validate_hadamard()}")
```

### Command Line

```bash
# Search for order 668 (default)
python hadamard_search.py

# Search for a different order
python -c "from hadamard_search import *; \
config = HadamardSearchConfig(matrix_order=100); \
searcher = HadamardSearcher(config); \
searcher.search()"
```

## Examples

### Finding a Known Hadamard Matrix (Order 16)

```python
from hadamard_search import HadamardSearcher, HadamardSearchConfig
import numpy as np

config = HadamardSearchConfig(matrix_order=16, max_restarts=5)
searcher = HadamardSearcher(config)
H, fitness = searcher.search()

print(f"Order 16 found with fitness: {fitness}")
# Expected: fitness = 0 (exact Hadamard matrix)

# Save to file
np.save('hadamard_16.npy', H.cpu().numpy())
```

### Validating a Matrix

```python
import numpy as np
from hadamard_search import HadamardSearcher, HadamardSearchConfig

# Load a matrix
H = np.load('hadamard_16.npy')

# Validate
config = HadamardSearchConfig(matrix_order=16)
searcher = HadamardSearcher(config)
is_valid = searcher.validate_hadamard(H)

print(f"Valid Hadamard: {is_valid}")
```

## Project Structure

```
hadamard-matrices/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusions
├── hadamard_search.py           # Main SA-GA search algorithm
└── examples/
    └── basic_search.py          # Usage examples
```

## Theory & Construction Methods

### Classical Constructions

1. **Sylvester-Hadamard (Recursive)**
   - Kronecker product: H₂ₙ = [Hₙ Hₙ; Hₙ -Hₙ]
   - Orders: all powers of 2

2. **Paley Construction**
   - Uses quadratic residues in finite fields
   - Orders: q ≡ 3 (mod 4) prime, gives order q+1
   - Example: q=11 → H₁₂

3. **Twin-Prime Construction**
   - Difference sets over F_p × F_q where p, q are twin primes
   - Orders: (p+1)(q+1)
   - Example: (2,3) → H₁₆

4. **Cyclotomy (Finite Field Extensions)**
   - Paley-type construction over F_{p^k}

## References

### Primary Sources

- **S. Markaida García** (2026). "Hadamard Matrices." BSc Thesis, UPV/EHU.
  - Supervisor: Luis Martínez Fernández
  - Covers theory, applications, constructions, and metaheuristic search

### Classical Literature

- Hadamard, J. (1893). "Résolution d'une question relative aux déterminants."
- Paley, R.E.A.C. (1933). "On orthogonal matrices." *J. Mathematics and Physics*.
- MacWilliams & Sloane (1977). *The Theory of Error-Correcting Codes*.
- Kharaghani & Tayfeh-Rezaie (2005, 2009). Hadamard classifications.

## Author

**Sara Markaida García**
- Email: saramarkaida10@gmail.com
- Location: London, UK
- Education: MSc Mathematics and Finance (Imperial College London, 2026–2027)

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this code in academic work, please cite:

```bibtex
@thesis{markaida2026hadamard,
  author    = {Markaida García, Sara},
  title     = {Hadamard Matrices},
  school    = {University of the Basque Country (UPV/EHU)},
  year      = {2026}
}
```

---

**Status**: Active research. Feedback welcome! 🚀
