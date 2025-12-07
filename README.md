# Basic Black Hole Simulator

This project is a Julia-based 3D orbital simulator for massive objects around a black hole. It's designed to simulate and compare orbits using both Newtonian gravity and a post-Newtonian approximation for a non-rotating (Schwarzschild) black hole.

## Features 

*   **3D Orbital Simulation:** Simulates trajectory of a particle around a central body.
*   **Two Physics Models:**
    *   **Newtonian Gravity:** Classical model of gravity.
    *   **Schwarzschild Approximation:** Post-Newtonian approximation for general relativity, suitable for non-rotating black holes.
*   **Event Horizon:** The simulation automatically stops if the particle crosses the event horizon of the black hole.
*   **Visualisation:** Generates plots and animations of the simulated orbits.

## Getting Started

### Prerequisites

*   Julia 

### Running the Simulations

1.  **Clone the repository.**
2.  **Navigate to the project directory.**
3.  **Activate the project environment:**
    ```bash
    julia --project=. -e "using Pkg; Pkg.instantiate()"
    ```

#### Compare Newtonian and Schwarzschild Gravity

To run a simulation that compares two gravity models and generates a plot:

```bash
julia scripts/compare_newton_schwarz.jl
```

#### Run a Single Simulation

To run a single simulation with the Schwarzschild model and generate a plot and an animation:

```bash
julia scripts/run_simulation
```
