module BasicBlackHoleSim

# External Packages
using DifferentialEquations
using Plots
using LinearAlgebra
using RecursiveArrayTools

# Project Structure 
include("Constants.jl")
include("Physics.jl")
include("Utils.jl")
include("Solvers.jl")

# Bring into scope if needed
using .Constants
using .Physics
using .Solvers
using .Utils

# User-end
export Constants                   # Access G, c
export get_black_hole_parameters   # Access helper
export circular_velocity           # Access helper
export calculate_circular_geodesic_4velocity # Access helper
export simulate_orbit              # Main
export plot_orbit, animate_orbit   # Visualisation


"""
High-level wrapper to run a simulation.
"""
function simulate_orbit(model_type, u0, tspan, M; kwargs...)
    prob = Solvers.setup_problem(model_type, u0, tspan, M)
    return Solvers.solve_orbit(prob; kwargs...)
end

end 