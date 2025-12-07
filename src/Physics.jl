module Physics

using ..Constants 
using LinearAlgebra 

# EXPORTS: We now export the split functions
export velocity_law!, newtonian_acceleration!, schwarzschild_acceleration!, kerr_acceleration!, kerr_geodesic!

"""
VELOCITY LAW
"""
function velocity_law!(dq, v, q, p, t)
    dq[1] = v[1]
    dq[2] = v[2]
    dq[3] = v[3]
end

"""
NEWTONIAN ACCELERATION
"""
function newtonian_acceleration!(dv, v, q, p, t)

    x, y, z = q[1], q[2], q[3]
    M = p
    r = norm(q)

    prefactor = -(Constants.G * M) / (r^3)
    
    # Update Acceleration (dv)
    dv[1] = prefactor * x
    dv[2] = prefactor * y
    dv[3] = prefactor * z
end

"""
SCHWARZSCHILD ACCELERATION
"""
function schwarzschild_acceleration!(dv, v, q, p, t)
    x, y, z = q[1], q[2], q[3]
    M = p
    r_sq = x^2 + y^2 + z^2
    r = sqrt(r_sq)
    
    h_vec = cross(q, v) 
    h_sq = dot(h_vec, h_vec) 

    # Coefficients
    # Matches the more efficient style in kerr_acceleration!
    term_newton = -(Constants.G * M) / (r * r_sq) # -GM/r^3
    term_gr = -(3 * Constants.G * M * h_sq) / (Constants.c^2 * r_sq^2 * r) # -3GMh^2/(c^2 r^5)

    total_coeff = term_newton + term_gr

    # Update Acceleration
    dv[1] = total_coeff * x
    dv[2] = total_coeff * y
    dv[3] = total_coeff * z 
end

"""
KERR ACCELERATION 
Includes Schwarzschild precession and Lense-Thirring frame-dragging.
"""
function kerr_acceleration!(dv, v, q, p, t)
    x, y, z = q[1], q[2], q[3]
    vx, vy, vz = v[1], v[2], v[3]
    M, a_star = p 
    r_sq = x^2 + y^2 + z^2
    r = sqrt(r_sq)

    # --- Schwarzschild part (central force) ---
    h_vec = cross(q, v) 
    h_sq = dot(h_vec, h_vec) 

    term_newton = -(Constants.G * M) / (r * r_sq) # -GM/r^3
    term_gr = -(3 * Constants.G * M * h_sq) / (Constants.c^2 * r_sq^2 * r) # -3GMh^2/(c^2 r^5)

    central_coeff = term_newton + term_gr

    ax_central = central_coeff * x
    ay_central = central_coeff * y
    az_central = central_coeff * z

    # --- Lense-Thirring part (frame-dragging, non-central force) ---
    # Assumes rotation is along the z-axis, only apply if black hole is spinning
    if a_star != 0.0

        # black hole's angular momentum
        J_mag = a_star * Constants.G * M^2 / Constants.c
        
        # Prefactor Lense-Thirring acceleration
        prefactor_lt = (2 * Constants.G * J_mag) / (Constants.c^2 * r * r_sq)
        
        # Lense-Thirring acceleration components
        ax_lt = prefactor_lt * ( (3 * z / r_sq) * h_vec[1] + vy )
        ay_lt = prefactor_lt * ( (3 * z / r_sq) * h_vec[2] - vx )
        az_lt = prefactor_lt * ( (3 * z / r_sq) * h_vec[3] )

        # Total acceleration
        dv[1] = ax_central + ax_lt
        dv[2] = ay_central + ay_lt
        dv[3] = az_central + az_lt
    else
        # If a_star is 0, just Schwarzschild
        dv[1] = ax_central
        dv[2] = ay_central
        dv[3] = az_central
    end
end

"""
Full geodesic equations for the Kerr metric.
Solves for the 8-component state vector u = [t, r, theta, phi, ut, ur, utheta, uphi]
"""
function kerr_geodesic!(du, u, p, λ)

    M, a = p
    
    # Unpack state vector
    t, r, theta, phi, ut, ur, utheta, uphi = u

    # Helper variables
    a2 = a^2
    r2 = r^2
    costheta = cos(theta)
    sintheta = sin(theta)
    cos2theta = costheta^2
    sin2theta = sintheta^2

    sigma = r2 + a2 * cos2theta
    lambda = r2 - 2 * M * r + a2
    
    # --- First 4 derivatives are just the 4-velocities ---
    du[1] = ut
    du[2] = ur
    du[3] = utheta
    du[4] = uphi

    # --- 4-accelerations d(u^μ)/dλ ---
    # Common terms
    sigma_inv = 1.0 / sigma
    lambda_inv = 1.0 / lambda
    
    # d(u^r)/dλ
    term_r1 = (M * (a2 * cos2theta - r2) * lambda_inv + r * a2 * sin2theta) * ut^2
    term_r2 = -2.0 * a * sin2theta * (M * (r2 - a2 * cos2theta) * lambda_inv + r * (sigma - M*r)) * ut * uphi
    term_r3 = -( (r - M) * lambda_inv - r * sigma_inv ) * ur^2 * sigma
    term_r4 = -r * lambda * utheta^2
    term_r5 = -(lambda * sin2theta - (2.0 * M * r * a2 * sin2theta^2 * sigma_inv)) * uphi^2
    du[6] = (term_r1 + term_r2 + term_r3 + term_r4 + term_r5) * sigma_inv
    
    # d(u^θ)/dλ
    du[7] = (sin(2*theta) * (a2 * ut^2 - 2*a*(r2+a2)*ut*uphi + (r2+a2)^2*uphi^2)/lambda - 2*a2*sin(2*theta)*utheta^2 - 4*r*ur*utheta) / (2*sigma)

    # d(u^φ)/dλ (must be calculated before d(u^t)/dλ)
    du[8] = (2/sigma) * ( (r-M)*ur*uphi + (r*uphi + a*ut)*ur + a*sin2theta*utheta*(ut-a*uphi) ) - (4*M*r*a*sin2theta/sigma^2)*utheta*ut

    # d(u^t)/dλ (depends on the du[8])
    du[5] = (2*M*r/sigma) * (ut*ur - a*sin2theta*ur*uphi) - (4*M*r*a*sin2theta/sigma)*utheta*du[8]

    return nothing
end

end