module Visuals

using Plots
using DifferentialEquations
using Printf
using ..Constants 

export plot_orbit, animate_orbit

"""
Plots the trajectory of the particle.
calculates the Event Horizon radius based 
"""
function plot_orbit(sol; title="Black Hole Orbit")
    x = sol[1, :]
    y = sol[2, :]
    
    # Extract Physics
    M = sol.prob.p
    Rs = (2 * Constants.G * M) / Constants.c^2

    # Create the Base Plot
    p = plot(x, y, 
             label="Trajectory", 
             xlabel="x (m)", 
             ylabel="y (m)", 
             title=title,
             linewidth=1.5,
             linecolor=:blue,
             aspect_ratio=:equal, 
             grid=true,
             framestyle=:box)

    # Draw the Black Hole (Event Horizon)
    theta = range(0, 2π, length=100)
    horizon_x = Rs .* cos.(theta)
    horizon_y = Rs .* sin.(theta)
    
    # Fill the event horizon with black
    plot!(p, horizon_x, horizon_y, seriestype=[:shape], 
          c=:black, fillalpha=1.0, linecolor=:black, 
          label="Event Horizon ($(@sprintf("%.2e", Rs)) m)")

    # Mark the start and end points
    scatter!(p, [x[1]], [y[1]], color=:green, label="Start")
    scatter!(p, [x[end]], [y[end]], color=:red, label="End")

    return p
end

"""
Creates a GIF animation of the orbit.
"""
function animate_orbit(sol, filename="orbit.gif"; fps=30)
    M = sol.prob.p
    Rs = (2 * Constants.G * M) / Constants.c^2
    
    step_size = max(1, floor(Int, length(sol.t) / 300)) 
    indices = 1:step_size:length(sol.t)

    anim = @animate for i in indices

        x_trail = sol[1, 1:i]
        y_trail = sol[2, 1:i]
        
        # Setup plot
        p = plot(x_trail, y_trail, 
                 xlims=(minimum(sol[1,:]), maximum(sol[1,:])),
                 ylims=(minimum(sol[2,:]), maximum(sol[2,:])),
                 aspect_ratio=:equal,
                 label="Path", linecolor=:blue, alpha=0.5)
                 
        # Draw Black Hole
        theta = range(0, 2π, length=50)
        plot!(p, Rs .* cos.(theta), Rs .* sin.(theta), 
              seriestype=[:shape], c=:black, label="Event Horizon")

        # Draw current position
        scatter!(p, [sol[1, i]], [sol[2, i]], color=:red, label="Particle", markersize=4)
        
        title!(p, "Time: $(round(sol.t[i], digits=2)) s")
    end

    gif(anim, filename, fps=fps)
    println("Animation saved to $filename")
end

end