 # To do
## Implementations:
* Modify optimization with respect of  SCF convergence
* ReadMolden (a way to star molecule for it).
* Fussion of viewer and DiffiQult.

# Backlog
* Implementations
  - Line profiler and computational graph analysis, find obvious tricks to optimize,
  - Documentation (comment the code)
  - Separate the code into folders and rename rutines.
  - Code review
  - Refactoring:
     - Check if you can get 2 gradients in one call feedind to UTMP variables in con jacobian extract.
* Test
  - Find unit tests for optimization
  - Add test for selected terms, use the same functions for grad and optimization for anyterm.
  - Unit test for grad of energy (re-write it).
* Future projects
   * MP2 (second)
     - Spectroscopy
   * Water
     - Cluster + electron UHF 
