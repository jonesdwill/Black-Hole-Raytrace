using Pkg

# Activate the project environment
projectdir(args...) = joinpath(@__DIR__, "..", args...)
Pkg.activate(projectdir()) 

using BasicBlackHoleSim
using BasicBlackHoleSim.Constants
using Plots

# plotlyjs() # Uncomment for interactive 3D plots

# --- CONFIG ---

M = 1.0 * M_sun
a_star = 0.98 # High spin parameter

# --- Initial Conditions ---
r0 = 5.0e7 # Starting radius (50,000 km), a much safer distance
v0 = circular_velocity(M, r0) * 1.1 # Start with 110% of circular velocity for a precessing ellipse
tspan = (0.0, 500.0) # Simulate for a few orbits
u0 = [r0, 0.0, 0.0,  0.0, v0, 0.1*v0] # Give a slight z-kick to see 3D frame-dragging

# --- SIMULATION ---

println("1. Simulating Orbit (Kerr Model, a*=$a_star)...")
sol = simulate_orbit(:kerr, u0, tspan, (M, a_star))

if sol.retcode == :Terminated
    println("IMPACT! The particle hit the Event Horizon.")
else
    println("Simulation finished safely. Steps: $(length(sol.t))")
end

# --- VISUALISATION ---

println("2. Generating Plot...")
p = plot_orbit(sol, title="Kerr Geodesic (a*=$a_star)")
display(p)
savefig(p, projectdir("scripts", "kerr_orbit_result.png"))
println("Plot saved to: $(projectdir("scripts", "kerr_orbit_result.png"))")

println("3. Generating Zoomed-in Plot...")
params = get_black_hole_parameters((M, a_star))
zoom_radius = 2 * params.rs # Zoom to twice the Schwarzschild radius
p_zoom = plot_orbit(sol, title="Kerr Geodesic (Zoomed In)", zoom_radius=zoom_radius)
display(p_zoom)
savefig(p_zoom, projectdir("scripts", "kerr_orbit_zoom_result.png"))
println("Zoomed plot saved to: $(projectdir("scripts", "kerr_orbit_zoom_result.png"))")

# --- ANIMATION ---
println("4. Generating Animation...")
animate_orbit(sol, projectdir("scripts", "kerr_orbit.gif"))