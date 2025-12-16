# ==============================================================================
#                           PAN/ZOOM RENDER SCRIPT
# 
# NOTE: Physics are calculated per-frame due to camera movement (pan/zoom).
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
using JLD2 # Still used, but mostly for saving the final animation
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
#                                 CONFIG
# ==============================================================================

const OUTPUT_DIR = @__DIR__ # output path

# config structure (simplified, no data_path needed as physics is inline)
struct Config
    width::Int
    height::Int
    M::Float64
    a_star::Float64
    duration::Float64
    fps::Int
    output_path::String

    # --- PAN/ZOOM CONTROL ---
    initial_fov::Float64
    final_fov::Float64
    initial_observer_angle_deg::Float64 # Observer angle (theta_obs)
    final_observer_angle_deg::Float64 
    initial_pan_angle::Float64         # Angle in the plane for panning (phi)
    final_pan_angle::Float64 
end

const CFG = Config(
    480,   # Width
    270,   # Height
    1.0,    # Mass (M)
    0.99,   # Spin (a*). High for max affect.
    4.0,    # Duration (e.g., 4 seconds for a slower pan)
    30,     # FPS
    joinpath(OUTPUT_DIR, "black_hole_pan_480p.gif"),

    # --- PAN/ZOOM KEYFRAMES ---
    18.0,   # FOV starts wide
    10.0,   # FOV zooms in to be tight
    80.0,   # Observer angle starts at 80 degrees (inclination)
    80.0,   # Observer angle remains constant (no vertical tilt)
    0.0,    # Pan starts centered
    5.0,    # Pan shifts right (by 5 degrees horizontal angle in the observer's frame)
)

# ==============================================================================
#                           ANIMATION LOGIC
# ==============================================================================

"""
Interpolates camera parameters based on the current frame index.
"""
function get_interpolated_camera_params(f::Int, total_frames::Int, cfg::Config)
    t_norm = (f - 1) / (total_frames - 1) # Normalized time t in [0, 1]
    
    # Linear interpolation function: start + t * (stop - start)
    current_fov = cfg.initial_fov + t_norm * (cfg.final_fov - cfg.initial_fov)
    current_obs_angle = cfg.initial_observer_angle_deg + t_norm * (cfg.final_observer_angle_deg - cfg.initial_observer_angle_deg)
    current_pan_angle = cfg.initial_pan_angle + t_norm * (cfg.final_pan_angle - cfg.initial_pan_angle)
    
    return current_fov, deg2rad(current_obs_angle), current_pan_angle
end

"""
Modified version of compute_physics that calculates geodesics for a single frame,
using dynamic FOV and camera pan.
"""
function compute_physics_for_frame(cfg::Config, fov_y::Float64, obs_angle_rad::Float64, pan_angle_deg::Float64)
    
    function calc_isco(a_star) 
        Z1 = 1.0 + cbrt(1.0 - a_star^2) * (cbrt(1.0 + a_star) + cbrt(1.0 - a_star))
        Z2 = sqrt(3.0 * a_star^2 + Z1^2)
        return 3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
    end

    r_isco = calc_isco(cfg.a_star) * cfg.M
    r_outer = 12.0 * cfg.M
    a = cfg.a_star * cfg.M
    r_horizon = cfg.M + sqrt(cfg.M^2 - a^2)
    
    data_r    = zeros(Float32, cfg.height, cfg.width)
    data_phi  = zeros(Float32, cfg.height, cfg.width)
    data_g    = zeros(Float32, cfg.height, cfg.width)
    data_mask = zeros(Bool, cfg.height, cfg.width)
    data_fell = zeros(Bool, cfg.height, cfg.width)
    
    aspect = cfg.width / cfg.height
    fov_x = fov_y * aspect # Use the frame-specific fov_y

    # Callbacks are the same as before
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
    
    # Pre-calculate the total angular offset for the horizontal pan
    pan_offset_rad = deg2rad(pan_angle_deg) 

    # No need for Threads.@threads here, as the main loop below is already threaded
    for i in 1:cfg.height
        beta = (i / cfg.height - 0.5) * 2 * fov_y # vertical angle, based on FOV
        
        for j in 1:cfg.width
            # Horizontal angle calculation includes the pan offset
            alpha = ((j - 0.5) / cfg.width - 0.5) * 2 * fov_x + pan_offset_rad 

            # The initial state function now requires the interpolated observer angle
            u0 = Utils.get_initial_photon_state_celestial(alpha, beta, 1000.0, obs_angle_rad, cfg.M, a)
            if any(isnan, u0) continue end

            prob = ODEProblem(Physics.kerr_geodesic_acceleration!, u0, (0.0, 2500.0), (cfg.M, a))
            # Lowered reltol/abstol for speed on pan, use 1e-6 for high quality 
            sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6, callback=disk_cb, save_everystep=false)    

            final_u = sol[end]
            r_hit = final_u[2]
            
            # Check for disk hit (same physics logic)
            hit_disk = abs(final_u[3] - π/2) < 0.01 && r_hit > r_isco && r_hit < r_outer
            
            if hit_disk
                try
                    ut_disk, uphi_disk = Utils.calculate_circular_orbit_properties(r_hit, cfg.M, a)
                    E_obs = -final_u[5]
                    E_emit = -(final_u[5] * ut_disk + final_u[8] * uphi_disk)
                    g = E_obs / E_emit
                    
                    if isfinite(g) && g > 0.1 && g < 3.0
                        data_mask[i, j] = true
                        data_r[i, j]    = Float32(r_hit)
                        data_phi[i, j]  = Float32(final_u[4])
                        data_g[i, j]    = Float32(g)
                    end
                catch e
                    # ignore error, mask is already false
                end
            else
                data_fell[i, j] = final_u[2] < r_horizon
            end
        end
    end
    
    return data_r, data_phi, data_g, data_mask, data_fell, r_isco, r_outer, r_horizon
end


# ==============================================================================
#                               MASTER RENDER LOOP
# ==============================================================================

function render_pan_animation(cfg::Config)

    println("\n=== STEP 1+2: RENDERING DYNAMIC PAN/ZOOM ANIMATION ===")
    
    total_frames = round(Int, cfg.duration * cfg.fps)
    println("Rendering $total_frames frames at $(cfg.width)x$(cfg.height)...")
    
    # --- NOISE SAMPLERS for TURBULENCE ---
    # These remain static/global across the animation
    noise_sampler_coarse = opensimplex2_4d(seed=2024)
    noise_sampler_medium = opensimplex2_4d(seed=2025)
    noise_sampler_fine = opensimplex2_4d(seed=2026)

    EXPOSURE = 1.5  # Global exposure
    render_collection = Vector{Array{RGB{N0f8}, 2}}(undef, total_frames) 
    
    p = Progress(total_frames, 1, "Tracing & Painting Frame...")

    # -------------- MASTER FRAME LOOP (Threaded over frames) --------------
    Threads.@threads for f in 1:total_frames 
        
        # 1. GET FRAME-SPECIFIC CAMERA PARAMETERS
        fov_y, obs_angle_rad, pan_angle_deg = get_interpolated_camera_params(f, total_frames, cfg)
        
        # 2. COMPUTE PHYSICS FOR THIS FRAME
        R, PHI, G, MASK, FELL, r_isco, r_outer, r_horizon = compute_physics_for_frame(
            cfg, fov_y, obs_angle_rad, pan_angle_deg
        )
        
        # 3. RENDER SHADER (mostly unchanged)
        
        time = (f-1) / cfg.fps
        local_frame_hdr = zeros(RGB{Float32}, cfg.height, cfg.width)

        M = cfg.M  # M and a_star must be accessible here
        a_star = cfg.a_star
        a = cfg.a_star * cfg.M

        for i in 1:cfg.height
            for j in 1:cfg.width
                
                # Background/Black Hole
                if !MASK[i, j]
                    local_frame_hdr[i, j] = RGB(0.0f0, 0.0f0, 0.0f0)
                    continue
                end
                
                r_hit = R[i, j]; phi_hit = PHI[i, j]; g = G[i, j]
                
                # --- SHADER EFFECTS ---
                Omega = 1.0 / (r_hit^1.5 + a_star)
                phi_unwrapped = phi_hit + Omega * time * 6.0                     
                r_norm = clamp((r_hit - r_isco) / (r_outer - r_isco), 0.0, 1.0)

                # --- TURBULENCE ---
                X_c = cos(phi_unwrapped) * r_hit * 0.3; Y_c = sin(phi_unwrapped) * r_hit * 0.3
                noise_coarse = sample(noise_sampler_coarse, X_c, Y_c, r_hit * 0.8, time * 0.4)

                X_m = cos(phi_unwrapped) * r_hit * 0.8; Y_m = sin(phi_unwrapped) * r_hit * 0.8
                noise_medium = sample(noise_sampler_medium, X_m, Y_m, r_hit * 2.5, time * 1.2)

                X_f = cos(phi_unwrapped) * r_hit * 1.5; Y_f = sin(phi_unwrapped) * r_hit * 1.5
                noise_fine = sample(noise_sampler_fine, X_f, Y_f, r_hit * 4.0, time * 2.5)

                combined_noise = (noise_coarse * 0.3) + (noise_medium * 0.5) + (noise_fine * 0.2)
                norm_noise = (combined_noise + 1.0) / 2.0
                turbulence = 0.4 + 0.6 * norm_noise^5.0

                spiral_phase = phi_unwrapped * 4.0 + r_hit * 0.15 
                spiral = 1.0 + 0.3 * sin(spiral_phase)
                turbulence *= spiral

                # --- COLOURING ---
                g_norm = clamp((g - 0.3) / (1.8 - 0.3), 0.0, 1.0)
                if g_norm < 0.25
                    t = g_norm / 0.25
                    base = RGB(0.5f0 + 0.3f0*t, 0.1f0 + 0.1f0*t, 0.05f0)
                elseif g_norm < 0.6
                    t = (g_norm - 0.25) / 0.35
                    base = RGB(0.8f0 + 0.2f0*t, 0.2f0 + 0.2f0*t, 0.05f0 + 0.05f0*t)
                else
                    t = (g_norm - 0.6) / 0.4
                    base = RGB(1.0f0, 0.5f0, 0.1f0 + 0.1f0*t)
                end
                
                # --- INTENSITY & OPACITY (with smooth doppler fix) ---
                g_clamped = clamp(g, 0.2, 3.0)
                g_squashed = g_clamped / (1.0 + g_clamped * 0.5) 
                doppler_boost = g_squashed^3.5 

                radial_falloff = 1.0 + 2.0 * (1.0 - r_norm)^2.0
                intensity = doppler_boost * radial_falloff * 2

                inner_fade = clamp((r_hit - r_isco) / 0.4, 0.0, 1.0)^3.5
                outer_fade = clamp((r_outer - r_hit) / 1.8, 0.0, 1.0)^3.5
                opacity = inner_fade * outer_fade
                opacity_final = opacity * turbulence

                hdr_color = base * intensity * opacity_final * 1.5
                local_frame_hdr[i, j] = hdr_color
            end
        end

        # --- POST-PROCESSING ---
        bright_threshold = 0.8
        bright_pass = map(c -> Gray(c) > bright_threshold ? c : RGB(0.0f0, 0.0f0, 0.0f0), local_frame_hdr)
        
        glow = imfilter(bright_pass, Kernel.gaussian(10.0))
        bloomed = local_frame_hdr .+ (glow .* 0.15) 
        
        # Tone mapping
        tone_mapped = map(bloomed) do c
            exposed_c = c * EXPOSURE
            RGB(
                red(exposed_c) / (1.0 + red(exposed_c)),
                green(exposed_c) / (1.0 + green(exposed_c)),
                blue(exposed_c) / (1.0 + blue(exposed_c))
            )
        end
        
        # Final clamp and storage
        final_8bit_frame = map(c -> RGB{N0f8}(
            clamp(red(c), 0, 1),
            clamp(green(c), 0, 1),
            clamp(blue(c), 0, 1)
        ), tone_mapped)

        render_collection[f] = final_8bit_frame
        next!(p) 
    end
    
    println("Saving to disk...")
    
    animation_stack = cat(render_collection..., dims=3)     
    save(cfg.output_path, animation_stack, fps=cfg.fps)     
    
    println("✓ Render complete: $(cfg.output_path)")
end

# ==============================================================================
#                           EXECUTION
# ==============================================================================

render_pan_animation(CFG)

println("\n=== COMPLETE ===")