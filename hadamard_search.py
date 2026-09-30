"""
Memetic Simulated Annealing + Genetic Algorithm for Hadamard Matrix Construction

Reference: Markaida García, S. (2026). Hadamard Matrices. Degree in Mathematics dissertation,
University of the Basque Country.

This module implements a hybrid metaheuristic combining Simulated Annealing (SA) with
Genetic Algorithm (GA) recombination to search for Hadamard matrices of arbitrary order,
with special focus on the open Hadamard Conjecture (order 668).
"""

import torch
import numpy as np
import time
from pathlib import Path


class HadamardSearchConfig:
    """Configuration parameters for the SA-GA search."""
    
    def __init__(
        self,
        matrix_order: int = 668,
        population_size: int = 96,
        max_iters_per_restart: int = 300_000,
        max_restarts: int = 60,
        ga_recombine_every: int = 3_000,
        final_temperature: float = 0.05,
        elite_fraction: float = 0.25,
        mutation_fraction: float = 0.50,
        mutation_rate: float = 0.01,
        device: str = None,
    ):
        """
        Initialize search configuration.
        
        Args:
            matrix_order: Dimension of Hadamard matrix to construct.
            population_size: Number of parallel SA chains (GA population).
            max_iters_per_restart: Single-bit flips per chain, per restart.
            max_restarts: Total number of restarts.
            ga_recombine_every: Recombine every K SA steps.
            final_temperature: Final temperature (near-greedy regime).
            elite_fraction: Fraction of elite individuals kept.
            mutation_fraction: Fraction of offspring that get mutated.
            mutation_rate: Per-entry bit-flip probability in mutation.
            device: 'cuda' or 'cpu' (auto-detects if None).
        """
        self.n = matrix_order
        self.population_size = population_size
        self.max_iters_per_restart = max_iters_per_restart
        self.max_restarts = max_restarts
        self.ga_recombine_every = ga_recombine_every
        self.final_temperature = final_temperature
        self.elite_fraction = elite_fraction
        self.mutation_fraction = mutation_fraction
        self.mutation_rate = mutation_rate
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.dtype = torch.float32


class HadamardSearcher:
    """Memetic SA-GA searcher for Hadamard matrices."""
    
    def __init__(self, config: HadamardSearchConfig):
        """Initialize searcher with configuration."""
        self.config = config
        self.n = config.n
        self.half = self.n // 2
        self.n_elite = max(2, int(config.population_size * config.elite_fraction))
        self.device = config.device
        self.dtype = config.dtype
        
        # Precompute off-diagonal mask for fitness calculation
        self.off_diag = ~torch.eye(self.n, dtype=torch.bool, device=self.device)
        
        self.best_fit = float('inf')
        self.best_matrix = None
        self.iteration_count = 0
    
    def _init_population(self, batch_size: int) -> torch.Tensor:
        """Initialize random ±1 population."""
        return (torch.randint(0, 2, (batch_size, self.n, self.n), 
                             device=self.device, dtype=self.dtype) * 2 - 1)
    
    def _build_gram(self, H: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix G = H @ H^T for batch of matrices."""
        return torch.bmm(H, H.transpose(1, 2))
    
    def _fitness_from_gram(self, G: torch.Tensor) -> torch.Tensor:
        """
        Compute fitness from Gram matrix.
        
        For a Hadamard matrix, G = nI (identity scaled by n).
        Fitness = sum_{i≠j} G_ij^2 = sum(G²) - n³
        Optimal fitness = 0 when H is Hadamard.
        """
        return (G * G).sum(dim=(1, 2)) - float(self.n) ** 3
    
    @torch.no_grad()
    def _sa_step(self, H: torch.Tensor, G: torch.Tensor, fit: torch.Tensor, 
                 T: float, batch_idx: torch.Tensor) -> None:
        """
        One Simulated Annealing step: random single-bit flip with Metropolis acceptance.
        
        Flip H[p, q]: only row/col p of G change via incremental update.
        dG[p, j] = -2*s*H[j, q] where s = H[p, q].
        """
        B = H.shape[0]
        p = torch.randint(0, self.n, (B,), device=self.device)
        q = torch.randint(0, self.n, (B,), device=self.device)
        
        s = H[batch_idx, p, q]  # (B,)
        
        # Extract column q and compute row change
        col_q = torch.gather(H, 2, q.view(B, 1, 1).expand(B, self.n, 1)).squeeze(2)
        dG_row = (-2.0) * s.view(B, 1) * col_q
        dG_row[batch_idx, p] = 0.0  # diagonal unchanged
        
        # Current and new row p of Gram matrix
        G_row = torch.gather(G, 1, p.view(B, 1, 1).expand(B, 1, self.n)).squeeze(1)
        new_G_row = G_row + dG_row
        dfit = 2.0 * (new_G_row * new_G_row - G_row * G_row).sum(1)
        
        # Metropolis acceptance
        prob = torch.exp(-torch.clamp(dfit, min=0.0) / T)
        acc = (dfit <= 0) | (torch.rand(B, device=self.device) < prob)
        
        # Update accepted moves
        bi = batch_idx[acc]
        H[bi, p[acc], q[acc]] = -s[acc]
        G[bi, p[acc], :] = new_G_row[acc]
        G[bi, :, p[acc]] = new_G_row[acc]  # Keep symmetric
        fit[acc] = fit[acc] + dfit[acc]
    
    @torch.no_grad()
    def _calibrate_temperature(self, H: torch.Tensor, G: torch.Tensor, 
                               fit: torch.Tensor, batch_idx: torch.Tensor) -> float:
        """Set T0 from typical uphill move so ~exp(-1/2) are accepted."""
        H_c, G_c, fit_c = H.clone(), G.clone(), fit.clone()
        self._sa_step(H_c, G_c, fit_c, T=1e9, batch_idx=batch_idx)
        d = fit_c - fit
        d = d[d > 0]
        return 2.0 * (d.mean().item() if d.numel() > 0 else 1.0)
    
    @torch.no_grad()
    def _crossover(self, elite: torch.Tensor, n_offspring: int) -> torch.Tensor:
        """Column-split crossover: child takes columns from each parent."""
        E = elite.shape[0]
        idx_f = torch.randint(0, E, (n_offspring,), device=self.device)
        idx_m = torch.randint(0, E, (n_offspring,), device=self.device)
        fathers, mothers = elite[idx_f], elite[idx_m]
        
        split = torch.randint(1, self.n, (n_offspring,), device=self.device)
        col_idx = torch.arange(self.n, device=self.device).view(1, 1, self.n)
        mask_m = col_idx >= split.view(n_offspring, 1, 1)
        return torch.where(mask_m, mothers, fathers)
    
    @torch.no_grad()
    def _mutate(self, pop: torch.Tensor, rate: float) -> torch.Tensor:
        """Bit-flip mutation."""
        flip = torch.rand_like(pop) < rate
        return torch.where(flip, -pop, pop)
    
    @torch.no_grad()
    def _ga_recombine(self, H: torch.Tensor, fit: torch.Tensor) -> torch.Tensor:
        """Keep elite, fill rest with mutated crossover offspring."""
        elite_idx = fit.topk(self.n_elite, largest=False).indices
        elite = H[elite_idx]
        n_off = self.config.population_size - self.n_elite
        
        offspring = self._crossover(elite, n_off)
        n_mut = max(1, int(n_off * self.config.mutation_fraction))
        mut_idx = torch.randperm(n_off, device=self.device)[:n_mut]
        offspring[mut_idx] = self._mutate(offspring[mut_idx], self.config.mutation_rate)
        
        return torch.cat([elite, offspring], dim=0)
    
    @torch.no_grad()
    def search(self, checkpoint_path: str = "hadamard_checkpoint.npy") -> tuple:
        """
        Run the hybrid SA-GA search.
        
        Returns:
            (best_matrix, best_fitness): Best Hadamard matrix found and its fitness.
        """
        checkpoint_path = Path(checkpoint_path)
        batch_idx = torch.arange(self.config.population_size, device=self.device)
        
        # Checkpoint recovery
        if checkpoint_path.exists():
            try:
                checkpoint = np.load(checkpoint_path)
                self.best_matrix = torch.tensor(checkpoint, dtype=self.dtype, device=self.device)
                G = self._build_gram(self.best_matrix.unsqueeze(0))
                self.best_fit = self._fitness_from_gram(G)[0].item()
                print(f"Checkpoint loaded - previous fitness: {self.best_fit:.0f}\n")
            except Exception as e:
                print(f"Warning: Could not load checkpoint ({e}). Starting fresh.\n")
        
        for restart in range(1, self.config.max_restarts + 1):
            H = self._init_population(self.config.population_size)
            G = self._build_gram(H)
            fit = self._fitness_from_gram(G)
            
            T0 = self._calibrate_temperature(H, G, fit, batch_idx)
            alpha = (self.config.final_temperature / T0) ** (1.0 / self.config.max_iters_per_restart)
            T = T0
            
            start_time = time.time()
            print(f"--- Restart {restart}/{self.config.max_restarts} (T0={T0:.3g} -> {self.config.final_temperature}) ---")
            
            for it in range(1, self.config.max_iters_per_restart + 1):
                self._sa_step(H, G, fit, T, batch_idx)
                T *= alpha
                self.iteration_count += 1
                
                # GA recombination
                if it % self.config.ga_recombine_every == 0:
                    H = self._ga_recombine(H, fit)
                    G = self._build_gram(H)
                    fit = self._fitness_from_gram(G)
                
                # Poll for new global best
                if it % 500 == 0:
                    curr_min = fit.min().item()
                    min_idx = fit.argmin()
                    if curr_min < self.best_fit:
                        self.best_fit = curr_min
                        self.best_matrix = H[min_idx].clone()
                        np.save(checkpoint_path, self.best_matrix.cpu().numpy())
                        if self.best_fit == 0.0:
                            print(f"\n*** SOLUTION FOUND: Restart {restart}, Step {it} ({time.time() - start_time:.1f}s) ***")
                            return self.best_matrix, self.best_fit
                
                # Resync fitness to kill numerical drift
                if it % 50_000 == 0:
                    fit = self._fitness_from_gram(G)
                
                # Logging
                if it % 20_000 == 0:
                    print(f"  Step {it:7d} | T {T:8.3g} | pop-min {fit.min().item():10.0f} | "
                          f"best {self.best_fit:10.0f} | {time.time() - start_time:5.1f}s")
            
            print(f"Restart {restart} done | global best {self.best_fit:.0f}")
            if self.best_fit == 0.0:
                break
        
        return self.best_matrix, self.best_fit
    
    def validate_hadamard(self, H: np.ndarray = None, tolerance: float = 1e-3) -> bool:
        """
        Validate if matrix is a valid Hadamard matrix.
        
        Args:
            H: Matrix to validate (if None, uses best found).
            tolerance: Numerical tolerance for H@H^T ≈ nI.
        
        Returns:
            True if H is a valid Hadamard matrix.
        """
        if H is None:
            H = self.best_matrix
            if H is None:
                return False
        
        if isinstance(H, np.ndarray):
            H = torch.tensor(H, dtype=self.dtype)
        
        H = H.to(self.device)
        HHt = H @ H.T
        n = H.shape[0]
        
        # Check diagonal = n
        diag_check = torch.allclose(HHt.diag(), torch.full((n,), float(n), device=self.device), 
                                    atol=tolerance)
        
        # Check off-diagonal ≈ 0
        off_diag_vals = HHt[self.off_diag]
        off_diag_check = torch.allclose(off_diag_vals, torch.zeros_like(off_diag_vals), 
                                       atol=tolerance)
        
        return diag_check and off_diag_check


def main():
    """Example usage: search for Hadamard matrix of order 668."""
    
    # Configuration
    config = HadamardSearchConfig(
        matrix_order=668,
        population_size=96,
        max_iters_per_restart=300_000,
        max_restarts=60,
        final_temperature=0.05,
    )
    
    print(f"Device: {config.device}")
    print(f"Matrix order: {config.n}×{config.n}")
    print(f"Population: {config.population_size}")
    print(f"Elite: {max(2, int(config.population_size * config.elite_fraction))}\n")
    
    # Create searcher and run
    searcher = HadamardSearcher(config)
    best_H, best_fit = searcher.search()
    
    # Results
    print(f"\nBest fitness for N={config.n}: {best_fit:.0f}")
    print(f"Status: {'EXACT Hadamard found!' if best_fit == 0 else 'Best approximation'}")
    
    if best_H is not None:
        # Save results
        mat = best_H.cpu().numpy().astype(int)
        np.save(f"hadamard_N{config.n}_fitness{int(best_fit)}.npy", mat)
        np.savetxt(f"hadamard_N{config.n}_fitness{int(best_fit)}.txt", mat, fmt="%d")
        
        # Validate
        is_valid = searcher.validate_hadamard(mat)
        print(f"Validation: {'✓ Valid Hadamard matrix' if is_valid else '✗ Not a Hadamard matrix'}")


if __name__ == "__main__":
    main()
