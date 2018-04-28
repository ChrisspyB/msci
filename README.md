# MSci Project
Code for simulating trajectories of massive and massless particles around a Kerr black hole in order to study the orbits of stars around the galactic centre.

# Basic Documentation

## List of files
- deriv_funcs_massive.py
	- Functions for integrating trajectories for massive particles.
- deriv_funcs_light.py
	- Functions for integrating trajectories for light.
- gillessen_orbits.txt
    - Data from Gillessen paper with orbital parameters around SgrA*.
- horiz_kerr_deflection.py
	- Calculates horizontal deflection (deflection of rays in the equatorial plane) of incident light rays for KBH. Plots:
        - deflection angle vs impact parameter w/ theoretical result.
        - y vs x ray trajectories.
- infall.py
    - Calculates radial proper infall time. Plots theoretical/simulated r against proper time.
- orbits.py
    - Calculate and many orbits using the gillensen data.  
- periodic_levin2008_kerr.py
	- Plot fig 15 (1,4,0) of Levin 2008.
- periodic_levin2008_schwarz.py
	- Plot fig 2 of Levin 2008.
- render.py
    - Produces VTK render of trajectories.
- s2.py
    - Calculate simulated/theoretical precession of S2 orbit. Plots:
        - Regular 3D Orbit in BH coords.
        - Orbit in orbital plane.
        - Orbit in Earth's sky.
- schwarz_precession.py
    - Calculate simulated and theoretical precession for SBH.
- schwarzschild_deflection.py
    - Calculate deflection of rays for SBH. Plots:
        - deflection angle vs impact parameter w/ theoretical result.
        - y vs x ray trajectories.
- tracer.py
    - For tracing rays around a central black hole.
- utils.py
    - Misc. collection of helper functions.
- tracing.nb
	- Mathematica notebook for verifying our maths. Specifically evaluating the nested derivatives in the ray tracing equations.
