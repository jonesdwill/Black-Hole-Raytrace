# ==============================================================================
#                      RENDER WRAPPER (COMPLETE SCRIPT)
# Module to render a black hole in two steps:
#     1. compute photon paths from grid of observer pixels and their redshifts.
#     2. render the accretion disk and lensed star background.
# ==============================================================================

# Activate Environment
using Pkg
projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir())

# Modules for rendering
using DifferentialEquations
using LinearAlgebra
using Base.Threads
using Images
using FileIO
using JLD2
using ProgressMeter
using Dates
using Printf
using ColorSchemes
using ImageFiltering
using CoherentNoise 
using Random 

# PHYSICS 
const SRC_DIR = joinpath(@__DIR__, "..", "..", "src")
println("Loading Physics modules from: $SRC_DIR")
include(joinpath(SRC_DIR, "Constants.jl"))
include(joinpath(SRC_DIR, "Physics.jl"))
include(joinpath(SRC_DIR, "Utils.jl"))

using .Constants
using .Physics
using .Utils

# ==============================================================================
#                                   CONFIG
# ==============================================================================

const OUTPUT_DIR = @__DIR__ # output path

# config structure 
struct Config
    width::Int
    height::Int
    M::Float64
    a_star::Float64
    fov_y::Float64
    duration::Float64
    fps::Int
    data_path::String
    output_path::String
end

const CFG = Config(
    1920,    # Width
    1080,    # Height
    1.0,     # Mass (M)
    0.99,    # Spin (a*). High for max affect.
    15.0,    # FOV (degrees)
    8,       # Duration 
    60,      # FPS
    joinpath(OUTPUT_DIR, "black_hole_data_1080p.jld2"),
    joinpath(OUTPUT_DIR, "black_hole_with_background_1080p.gif") 
)

# ==============================================================================
#                             COMPUTE PHYSICS (STEP 1)
# ==============================================================================

function compute_physics(cfg::Config)

    println("\n=== STEP 1: COMPUTING GEODESICS ===")
    println("Resolution: $(cfg.width)x$(cfg.height)")
    
    function calc_isco(a_star) 
        """Calculate the ISCO radius (in units of M) for a Kerr black hole with spin a* """
        Z1 = 1.0 + cbrt(1.0 - a_star^2) * (cbrt(1.0 + a_star) + cbrt(1.0 - a_star))
        Z2 = sqrt(3.0 * a_star^2 + Z1^2)
        return 3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
    end

    # Key Radii
    r_isco = calc_isco(cfg.a_star) * cfg.M       # ISCO radius
    r_outer = 12.0 * cfg.M                       # Outer radius
    a = cfg.a_star * cfg.M                       # Semi-major axis  
    r_horizon = cfg.M + sqrt(cfg.M^2 - a^2)      # Event horizon radius
    
    # Setup Buffers
    data_r        = zeros(Float32, cfg.height, cfg.width)
    data_phi      = zeros(Float32, cfg.height, cfg.width)
    data_g        = zeros(Float32, cfg.height, cfg.width)
    data_mask     = zeros(Bool, cfg.height, cfg.width)
    data_fell     = zeros(Bool, cfg.height, cfg.width)
    
    # --- NEW BUFFERS FOR BACKGROUND STAR COORDINATES ---
    data_theta_bg = zeros(Float32, cfg.height, cfg.width) 
    data_phi_bg   = zeros(Float32, cfg.height, cfg.width)
    # ----------------------------------------------------
    
    aspect = cfg.width / cfg.height      # Aspect ratio
    fov_x = cfg.fov_y * aspect           # Horizontal FOV


    # --- Set up ODE Callback to stop photons going through disc plane ---
    function disk_condition(u, t, integrator)
        return u[3] - π/2
    end

    function disk_affect!(integrator)
        r = integrator.u[2]
        if r > r_isco && r < r_outer
            terminate!(integrator)
        end
    end

    disk_cb = ContinuousCallback(disk_condition, disk_affect!)

    # --- Set up ODE Callback to stop photons crossing event horizon ---
    function horizon_condition(u, t, integrator)
        return u[2] - r_horizon
    end

    function horizon_affect!(integrator)
        terminate!(integrator)
    end

    horizon_cb = ContinuousCallback(horizon_condition, horizon_affect!)
    # --------------------------------------------------------------------

    p = Progress(cfg.height, 1, "Tracing Rays (Physics)...")     # Progress bar because this can take a while
    
    # -------------- MAIN LOOP OVER PIXELS --------------

    # loop over image rows
    Threads.@threads for i in 1:cfg.height 

        beta = (i / cfg.height - 0.5) * 2 * cfg.fov_y # get vertical angle
        
        # loop over image columns
        for j in 1:cfg.width

            alpha = ((j - 0.5) / cfg.width - 0.5) * 2 * fov_x # get horizontal angle

            # Spawn initial photon state from the observer coords
            u0 = Utils.get_initial_photon_state_celestial(alpha, beta, 1000.0, deg2rad(80.0), cfg.M, a)
            if any(isnan, u0) continue end

            # !!! SET-UP and SOLVE ODE !!!
            prob = ODEProblem(Physics.kerr_geodesic_acceleration!, u0, (0.0, 2500.0), (cfg.M, a))
            sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, callback=CallbackSet(disk_cb, horizon_cb), save_everystep=false)
            # !!! END ODE SOLVE !!!

            final_u = sol[end]  # Final state vector
            r_hit = final_u[2]  # Radius at disc intersection
            
            terminated = sol.retcode == :Terminated
            
            if terminated && abs(r_hit - r_horizon) < 1e-3
                # Terminated at horizon
                hit_disk = false
                data_fell[i, j] = true
            elseif terminated && r_hit > r_isco && r_hit < r_outer
                # Terminated at disk
                hit_disk = abs(final_u[3] - π/2) < 0.01
            else
                # Escaped or other (ray reached max time limit)
                hit_disk = false
                data_fell[i, j] = false
            end
            
            # Process Accretion Disk Hit
            if hit_disk
                try
                    # Calculate redshift factor g
                    ut_disk, uphi_disk = Utils.calculate_circular_orbit_properties(r_hit, cfg.M, a)
                    E_obs = -final_u[5]
                    E_emit = -(final_u[5] * ut_disk + final_u[8] * uphi_disk)
                    g = E_obs / E_emit
                    
                    if isfinite(g) && g > 0.1 && g < 3.0
                        data_mask[i, j] = true
                        data_r[i, j]    = Float32(r_hit)
                        data_phi[i, j]  = Float32(final_u[4])
                        data_g[i, j]    = Float32(g)
                    else
                        data_mask[i, j] = false
                    end

                # Catch any numerical errors and mark as invalid
                catch e
                    data_mask[i, j] = false
                end
            end
            
            # --- NEW: Process Escaped Ray (for background) ---
            if !hit_disk && !data_fell[i, j]
                # Ray escaped to infinity (or reached max time), save final angles
                # Final angle (theta) should be clamped to [0, pi]
                data_theta_bg[i, j] = Float32(clamp(final_u[3], 0.0, π))
                # Final angle (phi) is usually unwrapped, but we just need it's value mod 2pi
                data_phi_bg[i, j]   = Float32(final_u[4])
            end
            # --------------------------------------------------

        end
        next!(p)
    end
    
    # Save data
    println("\nSaving PHYSICS DATA to $(cfg.data_path)...")
    save(cfg.data_path, Dict(
        "r" => data_r,
        "phi" => data_phi,
        "g" => data_g,
        "mask" => data_mask,
        "fell" => data_fell,
        "theta_bg" => data_theta_bg, # <-- NEW
        "phi_bg" => data_phi_bg,     # <-- NEW
        "params" => (cfg.M, cfg.a_star, r_isco, r_outer, r_horizon)
    ))
end

# ==============================================================================
#                               RENDER ANIMATION (STEP 2)
# ==============================================================================

# Global variable to hold the background image
global STAR_BACKGROUND = nothing

function load_star_background()
    # Path to a star background image (e.g., Milky Way equirectangular projection)
    # **NOTE:** Update this path to your actual image file. 
    bg_path = projectdir("assets", "star_background.avif") 
    
    try
        # Try loading the actual image
        img = load(bg_path)
        global STAR_BACKGROUND = img
        println("Successfully loaded star background from: $bg_path")
    catch e
        # Fallback to a simple placeholder image if file not found
        println("WARNING: Star background file not found at $bg_path. Using a simple placeholder.")
        W, H = 2048, 1024 # A common equirectangular resolution
        dummy_img = [RGB{Float32}(0.1f0 + 0.3f0*cos(2π*j/W), 0.05f0, 0.2f0 + 0.3f0*sin(π*i/H)) for i in 1:H, j in 1:W]
        global STAR_BACKGROUND = dummy_img
    end
end


function render_animation(cfg::Config)

    println("\n=== STEP 2: RENDERING ANIMATION ===")
    
    # Check precomputed data exists
    if !isfile(cfg.data_path)
        println("ERROR: File not found! Run Step 1.")
        return
    end

    load_star_background() # Load the image once

    # Load precomputed data
    data = load(cfg.data_path)
    R = data["r"]; PHI = data["phi"]; G = data["g"]; MASK = data["mask"]; FELL = data["fell"]
    THETA_BG = data["theta_bg"]; PHI_BG = data["phi_bg"] # <-- NEW
    (M, a_star, r_isco, r_outer, r_horizon) = data["params"]
    
    # Calculate total frames needed for animation 
    frames = round(Int, cfg.duration * cfg.fps)
    println("Rendering $frames frames at $(cfg.width)x$(cfg.height)...")
    
    # FOV calculations
    aspect = cfg.width / cfg.height
    fov_x = cfg.fov_y * aspect
    
    # Get dimensions of background image
    H_bg, W_bg = size(STAR_BACKGROUND)

    # --- NOISE SAMPLERS for TURBULENCE ---
    noise_sampler_coarse = opensimplex2_4d(seed=2024)
    noise_sampler_medium = opensimplex2_4d(seed=2025)
    noise_sampler_fine = opensimplex2_4d(seed=2026)

    # --- POST-PROCESSING CONTROLS ---
    EXPOSURE = 1.5                                         # brightness factor
    render_collection = Vector{Array{RGB{N0f8}, 2}}(undef, frames) # Store rendered frames
    
    p = Progress(frames, 1, "Painting Disk...")


    # -------------- MAIN RENDER LOOP OVER FRAMES --------------
    Threads.@threads for f in 1:frames 

        # time in seconds
        time = (f-1) / cfg.fps
        
        # buffer just for this f-th frame
        local_frame_hdr = zeros(RGB{Float32}, cfg.height, cfg.width)

        # loop over image pixels
        for i in 1:cfg.height
            for j in 1:cfg.width

                # Check pixel mask and colour black if no disc hit
                if !MASK[i, j]

                    if FELL[i, j]
                        # Fell into black hole
                        local_frame_hdr[i, j] = RGB(0.0f0, 0.0f0, 0.0f0)
                    else
                        # Background - RAY ESCAPED TO INFINITY
                        # Get the final coordinates of the photon
                        final_theta = THETA_BG[i, j]
                        final_phi = PHI_BG[i, j]
                        
                        # Map polar angle (theta) from [0, pi] to [1, H_bg]
                        # Clamp is important to handle rays landing exactly at 0 or pi
                        y_map = round(Int, clamp(final_theta / π * H_bg, 1, H_bg)) 
                        
                        # Map azimuthal angle (phi) from [0, 2pi] to [1, W_bg]
                        phi_normalized = mod(final_phi, 2π) 
                        # Use (phi_normalized / 2π) * W_bg + 1 to map [0, 2pi] to [1, W_bg+1) then clamp
                        x_map = round(Int, clamp((phi_normalized / (2π)) * W_bg + 1, 1, W_bg)) 
                        
                        # Sample the background star image
                        star_color = STAR_BACKGROUND[y_map, x_map]
                        
                        # Set the pixel color
                        local_frame_hdr[i, j] = RGB{Float32}(star_color) * 1.5 # Boost brightness
                    end
                    continue
                end
                
                # --- ACCRETION DISK RENDERING LOGIC (The original shader) ---
                r_hit = R[i, j]          # Radius at disc hit
                phi_hit = PHI[i, j]      # Original azimuthal angle at disc hit
                g = G[i, j]              # Redshift factor
                
                # --------------- SHADER EFFECTS --------------

                Omega = 1.0 / (r_hit^1.5 + a_star)                 # Keplerian angular velocity
                phi_unwrapped = phi_hit + Omega * time * 6.0       # Unwrapped azimuthal angle at time t 
                r_norm = clamp((r_hit - r_isco) / (r_outer - r_isco), 0.0, 1.0) # Normalised radius [0, 1]

                # --------------- TURBULENCE ----------------
                X_c = cos(phi_unwrapped) * r_hit * 0.3
                Y_c = sin(phi_unwrapped) * r_hit * 0.3
                noise_coarse = sample(noise_sampler_coarse, X_c, Y_c, r_hit * 0.8, time * 0.4)

                X_m = cos(phi_unwrapped) * r_hit * 0.8
                Y_m = sin(phi_unwrapped) * r_hit * 0.8
                noise_medium = sample(noise_sampler_medium, X_m, Y_m, r_hit * 2.5, time * 1.2)

                X_f = cos(phi_unwrapped) * r_hit * 1.5
                Y_f = sin(phi_unwrapped) * r_hit * 1.5
                noise_fine = sample(noise_sampler_fine, X_f, Y_f, r_hit * 4.0, time * 2.5)
                # --------------------------------------------

                # Blend gaussian noise layers 
                combined_noise = (noise_coarse * 0.3) + (noise_medium * 0.5) + (noise_fine * 0.2)
                norm_noise = (combined_noise + 1.0) / 2.0
                turbulence = 0.4 + 0.6 * norm_noise^5.0

                # Add spiral pattern
                spiral_phase = phi_unwrapped * 4.0 + r_hit * 0.15 
                spiral = 1.0 + 0.3 * sin(spiral_phase)
                turbulence *= spiral

                # ---------- COLOURING ----------
                g_norm = clamp((g - 0.3) / (1.8 - 0.3), 0.0, 1.0)

                if g_norm < 0.25
                    t = g_norm / 0.25
                    base = RGB(0.6f0 + 0.3f0*t, 0.2f0 + 0.2f0*t, 0.05f0)
                elseif g_norm < 0.6
                    t = (g_norm - 0.25) / 0.35
                    base = RGB(0.9f0 + 0.1f0*t, 0.4f0 + 0.4f0*t, 0.05f0 + 0.15f0*t)
                else
                    t = (g_norm - 0.6) / 0.4
                    base = RGB(1.0f0, 0.8f0, 0.2f0 + 0.1f0*t)
                end
                # --------------------------------

                # ---------- INTENSITY & OPACITY ----------
                g_clamped = clamp(g, 0.2, 2.5)  
                doppler_boost = g_clamped^3.0

                radial_falloff = 1.0 + 2.0 * (1.0 - r_norm)^2.0
                intensity = doppler_boost * radial_falloff * 2

                inner_fade = clamp((r_hit - r_isco) / 0.4, 0.0, 1.0)^3.5
                outer_fade = clamp((r_outer - r_hit) / 1.8, 0.0, 1.0)^3.5
                opacity = inner_fade * outer_fade
                opacity_final = opacity * turbulence

                # final color of pixel 
                hdr_color = base * intensity * opacity_final * 1.5 
                local_frame_hdr[i, j] = hdr_color
                # -------------------------------------------
            end
        end

        # --------------- POST-PROCESSING (Applies to the full scene) --------------
        bright_threshold = 0.8
        bright_pass = map(c -> Gray(c) > bright_threshold ? c : RGB(0.0f0, 0.0f0, 0.0f0), local_frame_hdr)
        
        # Blooming 
        glow = imfilter(bright_pass, Kernel.gaussian(8.0))
        # Add the glow effect
        bloomed = local_frame_hdr .+ (glow .* 0.4) 
        
        # Tone mapping
        tone_mapped = map(bloomed) do c
            exposed_c = c * EXPOSURE
            RGB(
                red(exposed_c) / (1.0 + red(exposed_c)),
                green(exposed_c) / (1.0 + green(exposed_c)),
                blue(exposed_c) / (1.0 + blue(exposed_c))
                )
        end
        
        # Clamp colour and convert to 8-bit channel 
        final_8bit_frame = map(c -> RGB{N0f8}(
            clamp(red(c), 0, 1),
            clamp(green(c), 0, 1),
            clamp(blue(c), 0, 1)
        ), tone_mapped)

        render_collection[f] = final_8bit_frame
        next!(p) # update progress bar
        # --------------- END FRAME RENDER --------------
    end
    
    println("Saving to disk...")
    
    animation_stack = cat(render_collection..., dims=3)     # Stack frames into 3D array
    save(cfg.output_path, animation_stack, fps=cfg.fps)     # Save as GIF
    
    println("Render complete: $(cfg.output_path)")
end

# ==============================================================================
#                             EXECUTION SWITCH
# ==============================================================================

RUN_PHYSICS = false
RUN_RENDER  = true 

if RUN_PHYSICS
    compute_physics(CFG)
end

if RUN_RENDER
    render_animation(CFG)
end

println("\n=== COMPLETE ===")