using Pkg

# Activate the project environment
projectdir(args...) = joinpath(@__DIR__, "..", args...)
Pkg.activate(projectdir()) 

using BasicBlackHoleSim
using BasicBlackHoleSim.Constants
using Plots

# plotlyjs() # Uncomment for interactive 3D plot

# --- CONFIG ---

M_kg = 1.0 * M_sun
a_star = 0.98 # High spin

# --- Geometrized Units ---
M = (G * M_kg) / c^2 
a = a_star * M

# --- Initial Conditions: Choose ONE of the following sections ---

# --- 1. Innermost Stable Circular Orbit (ISCO) ---
# # should look like a simple, repeating circle.

# # println("Setting up ISCO simulation...")
# # r0 = 1.52 * M # Pre-calculated ISCO radius for a_star = 0.98
# # theta0 = π/2      # Equatorial plane
# # ur0 = 0.0
# # utheta0 = 0.0
# # ut0, uphi0 = calculate_circular_geodesic_4velocity(r0, theta0, M, a)
# # tspan = (0.0, 200.0 * M) 

# # --- 2. Tilted, Plunging Orbit (More Dramatic Visuals) ---
println("Setting up tilted, plunging orbit simulation...")

r0 = 6.0 * M
theta0 = π/2            # Start in the equatorial plane
ur0 = -0.0005           # Give it a very small inward push
utheta0 = 0.02          # Give it a kick

ut0, uphi0 = calculate_circular_geodesic_4velocity(r0, theta0, M, a)
tspan = (0.0, 300.0 * M)


# Assemble 8-component initial state vector
t0 = 0.0
phi0 = 0.0
u0 = [t0, r0, theta0, phi0, ut0, ur0, utheta0, uphi0]

# --- SIMULATION ---

println("1. Simulating Geodesic Orbit...")

sol = simulate_orbit(:kerr_geodesic, u0, tspan, (M, a), maxiters=1e7)

println("Simulation finished safely. Steps: $(length(sol.t))")

# --- VISUALISATION ---

println("2. Generating Plot...")
p = plot_orbit(sol, title="Tilted, Plunging Kerr Geodesic (a*=$a_star)")
display(p)
savefig(p, projectdir("scripts", "geodesic_orbit_result.png"))
println("Plot saved to: $(projectdir("scripts", "geodesic_orbit_result.png"))")

# # --- ANIMATION ---
# println("3. Generating Animation...")
# animate_orbit(sol, projectdir("scripts", "geodesic_orbit.gif"), num_animation_frames=200, max_trail_points=500) 