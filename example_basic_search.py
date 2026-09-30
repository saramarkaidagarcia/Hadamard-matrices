#!/usr/bin/env python3
"""
Simple example: Searching for a Hadamard matrix of order 16.

This example demonstrates basic usage of the HadamardSearcher.
Order 16 should be found quickly (exact Hadamard matrix exists).
"""

from hadamard_search import HadamardSearcher, HadamardSearchConfig
import numpy as np

# Configure search for order 16 (should find quickly)
config = HadamardSearchConfig(
    matrix_order=16,
    population_size=96,
    max_iters_per_restart=10_000,
    max_restarts=5,  # Reduced for demo
)

print("=" * 70)
print("Hadamard Matrix Search - Example (Order 16)")
print("=" * 70)
print(f"Device: {config.device}")
print(f"Matrix order: {config.n}×{config.n}")
print(f"Population size: {config.population_size}")
print()

# Create searcher
searcher = HadamardSearcher(config)

# Run search
best_matrix, best_fitness = searcher.search()

# Display results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Best fitness achieved: {best_fitness:.0f}")
print(f"Status: {'✓ EXACT Hadamard matrix found!' if best_fitness == 0 else '✗ Approximation'}")
print()

# Validate
if best_matrix is not None:
    is_valid = searcher.validate_hadamard()
    print(f"Validation: {'✓ Valid Hadamard matrix' if is_valid else '✗ Not valid'}")
    
    # Save
    mat = best_matrix.cpu().numpy().astype(int)
    filename = f"hadamard_H{config.n}_fitness{int(best_fitness)}.npy"
    np.save(filename, mat)
    print(f"Saved to: {filename}")
    
    # Display matrix (if small enough)
    if config.n <= 8:
        print("\nMatrix:")
        print(mat)

print("\n" + "=" * 70)
