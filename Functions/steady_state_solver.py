import sympy
from sympy import symbols, Eq, solve, exp

# Define variables (symbols)
Ca_in_SMC, Ca_SR, y = symbols('Ca_in_SMC Ca_SR y', positive=True, real=True)

def solve_initial_vars(params, init_vals):
    """Solve algebraic equations for Ca_in_SMC, Ca_SR, and y."""

    # map dictionary keys directly (unpack parameters?) with shorter parameter names (remove 'SMC_Par/')
    p = {k.split('/')[-1]: v for k, v in params.items()}

    # Equation 1
    eq1 = Eq(
        (
            
        (p['k_RyR']*(
            p['k_ryr0'] + (p['k_ryr1']*Ca_in_SMC**3 / (p['k_ryr2']**3 + Ca_in_SMC**3)))
                * (Ca_SR**4 / (p['k_ryr3']**4 + Ca_SR**4))
            + p['Jer'])*(Ca_SR-Ca_in_SMC)
        - (p['Ve']*Ca_in_SMC**2 / (p['Ke']**2 + Ca_in_SMC**2))
        + p['delta_SMC']*((
            p['alpha0']
            - p['alpha1']*(
                p['gca'] * (1/(1+exp(-(p['V0']-p['Vm'])/p['km'])))**2
                 *(p['V0']*(Ca_in_SMC - p['Ca_E']*exp(-2*p['V0']*p['F']/(p['R']*p['T'])))
                   / (1 - exp(-2*p['V0']*p['F']/(p['R']*p['T'])))
                )
            /(2*p['F']))
        )
        - (p['Vp']*Ca_in_SMC**4/(p['Kp']**4+Ca_in_SMC**4)))),
        0
        )

    # Equation 2
    eq2 = Eq(
        p['gamma']*(
            (p['Ve']*Ca_in_SMC**2 / (p['Ke']**2 + Ca_in_SMC**2))
            - ((p['k_RyR']*(
                p['k_ryr0'] + (p['k_ryr1']*Ca_in_SMC**3 / (p['k_ryr2']**3 + Ca_in_SMC**3)))
                    * (Ca_SR**4 / (p['k_ryr3']**4 + Ca_SR**4))              
                + p['Jer'])
              )*(Ca_SR-Ca_in_SMC)
        ),
        0
        )
    # Equation 3
    eq3 = Eq(
        ((p['l_4']* Ca_in_SMC)*(1-y) - p['l_m4']*y), 0)
    # Solve the system of equations symbolically

    y_expr = solve(eq3, y)[0]   # symbolic
    eq1_sub = eq1.subs(y, y_expr)
    eq2_sub = eq2.subs(y, y_expr)
    solutions = solve((eq1_sub, eq2_sub), (Ca_in_SMC, Ca_SR), dict=True)
    # Check if no solution
    if not solutions:
        print(" No symbolic solution found for initial variables.")
        print("Parameters causing failure:")
        for k, v in params.items():
            print(f"  {k}: {v}", end="  ")
        print()  # just to move to the next line after the loop

        # Second Optiona, using numerical solver (like nsolve)
        from sympy import nsolve, SympifyError
        try:
            numeric_sol = nsolve(
                (eq1, eq2, eq3), 
                (Ca_in_SMC, Ca_SR, y), 
                [init_vals['Ca_in_SMC'], init_vals['Ca_SR'], init_vals['y']],
                tol=1e-12,        # tight tolerance
                maxsteps=50       # maximum allowed iterations)
            )
            Ca_in_SMC_val, Ca_SR_val, y_val = [float(v) for v in numeric_sol]
            print(f"Numeric solution: Ca_in_SMC={Ca_in_SMC_val}, Ca_SR={Ca_SR_val}, y={y_val}")
        except Exception as e:
            # detect convergence failure specifically
            if "convergence" in str(e).lower() or "tolerance" in str(e).lower():
                raise RuntimeError(
               "Solver stopped: maximum steps or tolerance reached without convergence."
                ) from e
            else:
                raise  # re-raise other errors normally

    # If solutions exist, print all and return first
    print(f" Found {len(solutions)} symbolic solution(s):")
    for idx, sol in enumerate(solutions, start=1):
        print(f"Solution {idx}: Ca_in_SMC={sol[Ca_in_SMC]}, Ca_SR={sol[Ca_SR]}, y={sol[y]}")

    first_sol = solutions[0]
    Ca_in_SMC_val = float(first_sol[Ca_in_SMC])
    Ca_SR_val = float(first_sol[Ca_SR])
    y_val = float(first_sol[y])
    return Ca_in_SMC_val, Ca_SR_val, y_val

if False:
    # Equation 1
    eq1 = Eq(
        (
            (p['SMC_Par/k_ipr']*(
                (p['SMC_Par/p_agonist']*Ca_in_SMC*(1-y)) /
                ((p['SMC_Par/p_agonist']+p['SMC_Par/l_m1']/p['SMC_Par/l_1'])*(Ca_in_SMC+(p['SMC_Par/l_m5']/p['SMC_Par/l_5'])))
            )**3)
            + p['SMC_Par/k_RyR']*(
                (p['SMC_Par/k_ryr0'] + p['SMC_Par/k_ryr1']*Ca_in_SMC**3 / (p['SMC_Par/k_ryr2']**3 + Ca_in_SMC**3))
                * Ca_SR**4 / (p['SMC_Par/k_ryr3']**4 + Ca_SR**4)
            )
            + p['SMC_Par/Jer']
        )*(Ca_SR - Ca_in_SMC)
        - (p['SMC_Par/Ve']*Ca_in_SMC**2 / (p['SMC_Par/Ke']**2 + Ca_in_SMC**2))
        + p['SMC_Par/delta_SMC']*(
            p['SMC_Par/alpha0']
            - p['SMC_Par/alpha1']*(
                p['SMC_Par/gca'] * (1/(1+exp(-(p['SMC_Par/V0']-p['SMC_Par/Vm'])/p['SMC_Par/km'])))**2
                * (p['SMC_Par/V0']*(Ca_in_SMC - p['SMC_Par/Ca_E']*exp(-2*p['SMC_Par/V0']*p['SMC_Par/F']/(p['SMC_Par/R']*p['SMC_Par/T'])))
                   / (1 - exp(-2*p['SMC_Par/V0']*p['SMC_Par/F']/(p['SMC_Par/R']*p['SMC_Par/T'])))
                ) * 1
            )/(2*p['SMC_Par/F'])
            + p['SMC_Par/alpha2']*p['SMC_Par/p_agonist']
        )
        - (p['SMC_Par/Vp']*Ca_in_SMC**4/(p['SMC_Par/Kp']**4+Ca_in_SMC**4)),
        0
)

    # Equation 2
    eq2 = Eq(
        p['SMC_Par/gamma']*(
            (p['SMC_Par/Ve']*Ca_in_SMC**2 / (p['SMC_Par/Ke']**2 + Ca_in_SMC**2))
            - (
                (p['SMC_Par/k_ipr']*(
                    (p['SMC_Par/p_agonist']*Ca_in_SMC*(1-y)) /
                    ((p['SMC_Par/p_agonist']+p['SMC_Par/l_m1']/p['SMC_Par/l_1'])*(Ca_in_SMC+(p['SMC_Par/l_m5']/p['SMC_Par/l_5'])))
                )**3)
                + p['SMC_Par/k_RyR']*(
                    (p['SMC_Par/k_ryr0'] + p['SMC_Par/k_ryr1']*Ca_in_SMC**3 / (p['SMC_Par/k_ryr2']**3 + Ca_in_SMC**3))
                    * Ca_SR**4 / (p['SMC_Par/k_ryr3']**4 + Ca_SR**4)
                )
                + p['SMC_Par/Jer']
            )*(Ca_SR-Ca_in_SMC)
        ),
        0
    )

    if False:
        # Equation 3
        eq3 = Eq(
            (
                (p['SMC_Par/l_m4']*(p['SMC_Par/l_m2']/p['SMC_Par/l_2'])*(p['SMC_Par/l_m1']/p['SMC_Par/l_1'])
                + p['SMC_Par/l_m2']*(p['SMC_Par/l_m4']/p['SMC_Par/l_4'])*p['SMC_Par/p_agonist'])
                * Ca_in_SMC
                / ((p['SMC_Par/l_m4']/p['SMC_Par/l_4'])*(p['SMC_Par/l_m2']/p['SMC_Par/l_2'])*((p['SMC_Par/l_m1']/p['SMC_Par/l_1'])+p['SMC_Par/p_agonist']))
            )*(1-y)
            - (
                (p['SMC_Par/l_m2']*p['SMC_Par/p_agonist'] + p['SMC_Par/l_m4']*(p['SMC_Par/l_m3']/p['SMC_Par/l_3']))
                /((p['SMC_Par/l_m3']/p['SMC_Par/l_3'])+p['SMC_Par/p_agonist'])
            )*y,
            0
        )
