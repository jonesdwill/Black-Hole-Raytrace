# ==============================================================================
# RENDER WRAPPER: Kerr Black Hole Visualisation 
# ==============================================================================

using Pkg
projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir())

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

# --- IMPORT YOUR ACTUAL PHYSICS ---
const SRC_DIR = joinpath(@__DIR__, "..", "..", "src")
println("Loading Physics modules from: $SRC_DIR")
include(joinpath(SRC_DIR, "Constants.jl"))
include(joinpath(SRC_DIR, "Physics.jl"))
include(joinpath(SRC_DIR, "Utils.jl"))

using .Constants
using .Physics
using .Utils

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
const OUTPUT_DIR = @__DIR__

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
    1920,   # Width
    1080,   # Height
    1.0,    # Mass (M)
    0.99,   # Spin (a*)
    15.0,   # FOV (degrees)
    1.0/60.0,    # Duration (single frame)
    60,     # FPS
    joinpath(OUTPUT_DIR, "black_hole_data_1080p.jld2"),
    joinpath(OUTPUT_DIR, "black_hole_1080p_frame-2.png")  # PNG for single frame
)

# ==============================================================================
# STEP 1: COMPUTE PHYSICS - WITH ARTIFACT FIXES
# ==============================================================================
function step1_compute_physics(cfg::Config)
    println("\n=== STEP 1: COMPUTING GEODESICS (HIGH RES) ===")
    println("Resolution: $(cfg.width)x$(cfg.height)")
    
    function calc_isco(a_star) 
        Z1 = 1.0 + cbrt(1.0 - a_star^2) * (cbrt(1.0 + a_star) + cbrt(1.0 - a_star))
        Z2 = sqrt(3.0 * a_star^2 + Z1^2)
        return 3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
    end

    r_isco = calc_isco(cfg.a_star) * cfg.M
    r_outer = 12.0 * cfg.M
    a = cfg.a_star * cfg.M
    r_horizon = cfg.M + sqrt(cfg.M^2 - a^2)
    
    # Setup Buffers
    data_r    = zeros(Float32, cfg.height, cfg.width)
    data_phi  = zeros(Float32, cfg.height, cfg.width)
    data_g    = zeros(Float32, cfg.height, cfg.width)
    data_mask = zeros(Bool, cfg.height, cfg.width)
    data_fell = zeros(Bool, cfg.height, cfg.width)
    
    aspect = cfg.width / cfg.height
    fov_x = cfg.fov_y * aspect

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

    p = Progress(cfg.height, 1, "Tracing Rays (Physics)...")
    
    Threads.@threads for i in 1:cfg.height
        beta = (i / cfg.height - 0.5) * 2 * cfg.fov_y
        
        for j in 1:cfg.width
            alpha = ((j - 0.5) / cfg.width - 0.5) * 2 * fov_x

            u0 = Utils.get_initial_photon_state_celestial(alpha, beta, 1000.0, deg2rad(80.0), cfg.M, a)
            if any(isnan, u0) continue end

            prob = ODEProblem(Physics.kerr_geodesic_acceleration!, u0, (0.0, 2500.0), (cfg.M, a))
            
            ### ARTIFACT FIX: Increased ODE tolerances for stable g-factor ###
            # Tighter tolerances improve the stability near disk edges
            sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6, callback=disk_cb, save_everystep=false)
            
            final_u = sol[end]
            r_hit = final_u[2]
            
            # ARTIFACT FIX: Tighter theta tolerance to avoid grazing rays
            # The 0.05 tolerance was catching rays at very shallow angles
            hit_disk = abs(final_u[3] - π/2) < 0.01 && r_hit > r_isco && r_hit < r_outer
            
            if hit_disk
                try
                    ut_disk, uphi_disk = Utils.calculate_circular_orbit_properties(r_hit, cfg.M, a)
                    E_obs = -final_u[5]
                    E_emit = -(final_u[5] * ut_disk + final_u[8] * uphi_disk)
                    g = E_obs / E_emit
                    
                    # ARTIFACT FIX: Filter out extreme/invalid g-factors
                    # These occur from grazing rays and numerical instabilities
                    if isfinite(g) && g > 0.1 && g < 3.0
                        data_mask[i, j] = true
                        data_r[i, j]    = Float32(r_hit)
                        data_phi[i, j]  = Float32(final_u[4])
                        data_g[i, j]    = Float32(g)
                    else
                        data_mask[i, j] = false
                    end
                catch e
                    data_mask[i, j] = false
                end
            else
                data_fell[i, j] = final_u[2] < r_horizon
            end
        end
        next!(p)
    end
    
    println("\nSaving PHYSICS DATA to $(cfg.data_path)...")
    save(cfg.data_path, Dict(
        "r" => data_r,
        "phi" => data_phi,
        "g" => data_g,
        "mask" => data_mask,
        "fell" => data_fell,
        "params" => (cfg.M, cfg.a_star, r_isco, r_outer, r_horizon)
    ))
end

# ==============================================================================
# STEP 2: RENDER ANIMATION - FIXED ARTIFACTS VERSION
# ==============================================================================
function step2_render_animation(cfg::Config)
    println("\n=== STEP 2: RENDERING ANIMATION (ARTIFACTS FIXED) ===")
    
    if !isfile(cfg.data_path)
        println("ERROR: File not found! Run Step 1.")
        return
    end

    data = load(cfg.data_path)
    R = data["r"]; PHI = data["phi"]; G = data["g"]; MASK = data["mask"]; FELL = data["fell"]
    (M, a_star, r_isco, r_outer, r_horizon) = data["params"]
    
    frames = round(Int, cfg.duration * cfg.fps)
    println("Rendering $frames frames at $(cfg.width)x$(cfg.height)...")
    
    aspect = cfg.width / cfg.height
    fov_x = cfg.fov_y * aspect
    
    # Initialize noise generators
    noise_sampler_coarse = opensimplex2_4d(seed=2024)
    noise_sampler_medium = opensimplex2_4d(seed=2025)
    noise_sampler_fine = opensimplex2_4d(seed=2026)

    # --- POST-PROCESSING CONTROLS ---
    EXPOSURE = 2.5  # Multiplies brightness before tone mapping to lift mid-tones

    render_collection = Vector{Array{RGB{N0f8}, 2}}(undef, frames)
    
    p = Progress(frames, 1, "Painting Disk...")

    Threads.@threads for f in 1:frames
        time = (f-1) / cfg.fps
        
        local_frame_hdr = zeros(RGB{Float32}, cfg.height, cfg.width)

        for i in 1:cfg.height
            for j in 1:cfg.width
                alpha = ((j - 0.5) / cfg.width - 0.5) * 2 * fov_x
                beta = (i / cfg.height - 0.5) * 2 * cfg.fov_y
                
                if !MASK[i, j]
                    if FELL[i, j]
                        local_frame_hdr[i, j] = RGB(0.0f0, 0.0f0, 0.0f0)
                    else
                        # Pure black background
                        local_frame_hdr[i, j] = RGB(0.0f0, 0.0f0, 0.0f0)
                    end
                    continue
                end
                
                r_hit = R[i, j]
                phi_hit = PHI[i, j] # Keep the original, unwrapped value from physics
                g = G[i, j]
                
                # === FIXED SHADER: No artifacts ===

                # Disk rotation
                Omega = 1.0 / (r_hit^1.5 + a_star)
                
                # Calculate the continuous, UNWRAPPED angle
                phi_unwrapped = phi_hit + Omega * time * 6.0
                
                # === MULTI-SCALE STRUCTURE with continuous coordinates ===
                r_norm = clamp((r_hit - r_isco) / (r_outer - r_isco), 0.0, 1.0)

                # ARTIFACT FIX: Use the unwrapped angle for noise coordinates.
                # Using cos/sin on the unwrapped angle is equivalent to using a wrapped
                # angle, but it's simpler and avoids any potential numerical issues with
                # the mod() function that could cause a seam. This makes the noise
                # calculation consistent with the spiral arm calculation.
                X_c = cos(phi_unwrapped) * r_hit * 0.3
                Y_c = sin(phi_unwrapped) * r_hit * 0.3
                noise_coarse = sample(noise_sampler_coarse, X_c, Y_c, r_hit * 0.8, time * 0.4)

                X_m = cos(phi_unwrapped) * r_hit * 0.8
                Y_m = sin(phi_unwrapped) * r_hit * 0.8
                noise_medium = sample(noise_sampler_medium, X_m, Y_m, r_hit * 2.5, time * 1.2)

                X_f = cos(phi_unwrapped) * r_hit * 1.5
                Y_f = sin(phi_unwrapped) * r_hit * 1.5
                noise_fine = sample(noise_sampler_fine, X_f, Y_f, r_hit * 4.0, time * 2.5)

                # Blend with emphasis on medium scale (visible structure)
                combined_noise = (noise_coarse * 0.3) + (noise_medium * 0.5) + (noise_fine * 0.2)
                norm_noise = (combined_noise + 1.0) / 2.0

                # MODERATE contrast (not too extreme, not too smooth)
                # Increased exponent for higher contrast in the disk texture
                turbulence = 0.4 + 0.6 * norm_noise^5.0

                # Add subtle spiral pattern with continuous function
                ### ARTIFACT FIX: Vertical Black Line ###
                # Use the UNWRAPPED angle for the phase to prevent the pattern from
                # abruptly jumping at the 2π boundary, which causes the sharp line.
                spiral_phase = phi_unwrapped * 4.0 + r_hit * 0.15 
                spiral = 1.0 + 0.3 * sin(spiral_phase)
                turbulence *= spiral

                # === BALANCED COLOR: Warm but not overwhelming ===
                g_norm = clamp((g - 0.3) / (1.8 - 0.3), 0.0, 1.0)

                if g_norm < 0.25
                    # Cool outer regions: deep orange
                    t = g_norm / 0.25
                    base = RGB(0.6f0 + 0.3f0*t, 0.2f0 + 0.2f0*t, 0.05f0)
                elseif g_norm < 0.6
                    # Mid regions: bright orange-yellow
                    t = (g_norm - 0.25) / 0.35
                    base = RGB(0.9f0 + 0.1f0*t, 0.4f0 + 0.4f0*t, 0.05f0 + 0.15f0*t)
                else
                    # Hot inner: bright yellow-white (FIXED to be continuous)
                    t = (g_norm - 0.6) / 0.4
                    # This starts at the previous segment's endpoint: RGB(1.0, 0.8, 0.2)
                    base = RGB(1.0f0, 0.8f0, 0.2f0 + 0.1f0*t)
                end

                # ARTIFACT TEST: Use a known-smooth color scheme to eliminate the
                # hand-coded gradient as a source of banding.
                #base = get(ColorSchemes.hot, g_norm)

                # === BALANCED INTENSITY ===
                # ARTIFACT FIX: Clamp g-factor to prevent extreme brightening
                # Even if physics passed validation, rendering can amplify edge cases
                g_clamped = clamp(g, 0.2, 2.5)
                # Increased exponent for higher contrast between approaching/receding sides
                doppler_boost = g_clamped^3.0

                # Moderate inner brightening
                radial_falloff = 1.0 + 2.0 * (1.0 - r_norm)^2.0

                intensity = doppler_boost * radial_falloff * 2

                # === DISK PROFILE with STRUCTURE ===
                # Sharper inner edge by reducing the fade distance
                inner_fade = clamp((r_hit - r_isco) / 0.4, 0.0, 1.0)^3.5

                # Sharper outer fade by reducing the fade distance
                outer_fade = clamp((r_outer - r_hit) / 1.8, 0.0, 1.0)^3.5

                opacity = inner_fade * outer_fade

                # Keep turbulent gaps visible
                opacity_final = opacity * turbulence

                # === FINAL COLOR ===
                hdr_color = base * intensity * opacity_final * 1.5 # Increased final brightness
                
                local_frame_hdr[i, j] = hdr_color
            end
        end

        # === POST-PROCESSING: Minimal bloom ===
        bright_threshold = 0.8
        bright_pass = map(c -> Gray(c) > bright_threshold ? c : RGB(0.0f0, 0.0f0, 0.0f0), local_frame_hdr)
        
        # Smaller bloom spread
        glow = imfilter(bright_pass, Kernel.gaussian(8.0))
        
        # Much reduced bloom contribution
        bloomed = local_frame_hdr .+ (glow .* 0.4)
        
        # Mask out bloom from shadow to prevent artifacts at horizon
        for i in 1:cfg.height
            for j in 1:cfg.width
                if FELL[i, j]
                    bloomed[i, j] = RGB(0.0f0, 0.0f0, 0.0f0)
                end
            end
        end # Reduced bloom to decrease blurriness
        
        # === GENTLE TONE MAPPING ===
        tone_mapped = map(bloomed) do c
            # Apply exposure control to brighten the image before compression
            exposed_c = c * EXPOSURE
            RGB(
                # ARTIFACT FIX: Use a correct tone mapping operator.
                # The previous curve could produce values > 1.0, which were then
                # clamped to white, creating the band. This is the standard Reinhard
                # operator, which maps all positive numbers to the [0, 1) range,
                # correctly compressing the extreme highlights into a smooth gradient.
                red(exposed_c) / (1.0 + red(exposed_c)),
                green(exposed_c) / (1.0 + green(exposed_c)),
                blue(exposed_c) / (1.0 + blue(exposed_c))
            )
        end
        
        # === CLAMP AND STORE ===
        final_8bit_frame = map(c -> RGB{N0f8}(
            clamp(red(c), 0, 1),
            clamp(green(c), 0, 1),
            clamp(blue(c), 0, 1)
        ), tone_mapped)

        render_collection[f] = final_8bit_frame
        next!(p)
    end
    
    println("Saving to disk...")
    
    save(cfg.output_path, render_collection[1])
    
    println("✓ Render complete: $(cfg.output_path)")
end

# ==============================================================================
#                               EXECUTION
# ==============================================================================

# IMPORTANT: Set RUN_PHYSICS to true to regenerate data with artifact fixes!
RUN_PHYSICS = false  # Set to 'true' only if you need to re-calculate the physics data.
RUN_RENDER  = true   

if RUN_PHYSICS
    step1_compute_physics(CFG)
end

if RUN_RENDER
    step2_render_animation(CFG)
end

println("\n=== COMPLETE ===")