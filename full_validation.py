import control
import numpy as np

# 1. Damped Oscillator (Second Order)
m, k, c = 1.0, 10.0, 0.5
A1 = np.array([[0.0, 1.0], [-k/m, -c/m]])
B1 = np.array([[0.0], [1/m]])
C1 = np.array([[1.0, 0.0]])
D1 = np.array([[0.0]])
sys1 = control.ss(A1, B1, C1, D1)

# 2. 4th Order System (Butterworth Filter)
# Coefficients for s^4 + 2.613s^3 + 3.414s^2 + 2.613s + 1
sys2 = control.tf2ss(control.tf([1], [1, 2.6131, 3.4142, 2.6131, 1]))

systems = [("Oscillator", sys1), ("Butterworth4", sys2)]

print("--- PYTHON VALIDATION ---")
for name, sys in systems:
    print(f"\nSystem: {name}")
    print(f"Poles: {sys.poles()}")
    t = np.linspace(0, 10.0, 101)
    _, y = control.step_response(sys, T=t)
    print(f"Final Output (t=10s): {y[-1]:.6f}")
    # DC Gain
    print(f"DC Gain: {control.dcgain(sys):.6f}")
