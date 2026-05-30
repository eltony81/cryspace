# CrySpace

**CrySpace** is a powerful control systems library for the Crystal programming language, inspired by the Python Control Systems Library (`python-control`). It provides tools for the analysis and design of feedback control systems, leveraging [num.cr](https://github.com/crystal-data/num.cr) for high-performance linear algebra.

## Features

- **State-Space Systems**: Create and manipulate LTI (Linear Time-Invariant) systems in state-space form ($\dot{x} = Ax + Bu, y = Cx + Du$).
- **Transfer Functions**: Represent systems as ratios of polynomials.
- **System Interconnections**:
  - Parallel connection (`+`)
  - Series connection (`*`)
  - Feedback connection (`feedback`)
- **Stability Analysis**: Calculate system poles.
- **Discretization**: Convert continuous-time systems to discrete-time using Zero-Order Hold (ZOH).
- **Time Response**:
  - Step response simulation.
  - General ODE solvers (**Euler** and **Runge-Kutta 4**).
- **SISO & MIMO**: Support for Single-Input Single-Output and Multi-Input Multi-Output systems.

## Installation

1. Add the dependency to your `shard.yml`:

   ```yaml
   dependencies:
     cryspace:
       github: eltony81/cryspace
   ```

2. Install system dependencies (LAPACK and BLAS/CBLAS):
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install liblapack-dev libcblas-dev
   ```

3. Run `shards install`

## Usage

### 1. Classic Example: Mass-Spring-Damper System
A classic example of a state-space model is a **Mass-Spring-Damper system** (like a car suspension). It converts a second-order physical equation into a 2-state matrix system.

**The Physical Setup:**
Newton's Second Law for a mass $m$, spring stiffness $k$, and damping coefficient $c$ with an external force $u$:
```math
m\ddot{y} + c\dot{y} + ky = u
```

**State Definitions:**
- $x_1 = y$ (Position)
- $x_2 = \dot{y}$ (Velocity)

**State-Space Matrices:**
```math
A = \begin{bmatrix} 0 & 1 \\ -k/m & -c/m \end{bmatrix}
```
```math
B = \begin{bmatrix} 0 \\ 1/m \end{bmatrix}
```
```math
C = \begin{bmatrix} 1 & 0 \end{bmatrix} \text{ (Measuring position)}
```
```math
D = \begin{bmatrix} 0 \end{bmatrix}
```

**Implementation in CrySpace:**
```crystal
require "cryspace"

m, k, c = 1.0, 10.0, 0.5

a = [[0.0, 1.0], [-k/m, -c/m]].to_tensor
b = [[0.0], [1/m]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor

sys = CrySpace::StateSpace.new(a, b, c, d)
puts "System Poles: #{sys.poles}"
```

### 2. Electrical Example: RLC Circuit
An RLC circuit is a second-order system composed of a Resistor $R$, Inductor $L$, and Capacitor $C$.

**The Physical Equations:**
Using Kirchhoff's Voltage Law (KVL):
```math
L\frac{di_L}{dt} + Ri_L + v_C = u(t)
```
```math
C\frac{dv_C}{dt} = i_L
```

**State Definitions:**
- $x_1 = v_C$ (Capacitor Voltage)
- $x_2 = i_L$ (Inductor Current)

**State-Space Matrices:**
```math
A = \begin{bmatrix} 0 & 1/C \\ -1/L & -R/L \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 1/L \end{bmatrix}
```
```math
C = \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad D = \begin{bmatrix} 0 \end{bmatrix}
```

**Implementation in CrySpace:**
```crystal
R, L, C = 10.0, 1.0, 0.1

a = [[0.0, 1/C], [-1/L, -R/L]].to_tensor
b = [[0.0], [1/L]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor

rlc_sys = CrySpace::StateSpace.new(a, b, c, d)

# Analyze stability and calculate step response
puts "Is RLC stable? #{rlc_sys.is_stable?}"
t, x, y = rlc_sys.step_response(n_steps: 100)
puts "Final Capacitor Voltage: #{y.last[0, 0].value}"
```

### 3. Feedback Connection
```crystal
# Closed loop with unity gain feedback
k_gain = [[1.0]].to_tensor
sys_cl = sys.feedback(k_gain)
puts "Closed loop poles: #{sys_cl.poles}"
```

### 3. Step Response and State Analysis
You can simulate the system's response to a step input and obtain the trajectory of all internal states ($x$) and outputs ($y$).

#### Manual Iteration (Detailed)
```crystal
# Simulation for 5 seconds (50 steps of 0.1s)
t, x, y = sys.step_response(n_steps: 50)

puts "Time (s) | Position (Output) | Velocity (State x2)"
puts "-" * 55
t.each_with_index do |time, i|
  x2 = x[i][1, 0].value
  output = y[i][0, 0].value
  puts "#{time.round(2).to_s.ljust(8)} | #{output.round(4).to_s.ljust(17)} | #{x2.round(4)}"
end
```

#### Vectorized Approach (Convenient)
If you prefer to work with full matrices/tensors (similar to NumPy/MATLAB) without manual loops, you can pass a time vector:

```crystal
# Create a time vector from 0 to 10s with 0.1s steps
t = Float64Tensor.linear_space(0.0, 10.0, 101)

# Simulate returns 2D Tensors:
# states: [time_steps x n_states]
# outputs: [time_steps x n_outputs]
times, states, outputs = sys.simulate(t)

# Now you have all position values in a single column
positions = states[..., 0] 
velocities = states[..., 1]
system_outputs = outputs[..., 0]

puts "Final position: #{positions[-1].value}"
```

### 4. General ODE Solving (RK4)
You can solve arbitrary ODEs of the form:
```math
\dot{x} = f(x, t)
```
and calculate derived outputs.

#### Vectorized Approach (Convenient)
Passing a time vector to the solver returns results as Tensors, allowing for easy slicing and matrix operations:

```crystal
f = ->(x : Float64Tensor, t : Float64) {
  # Example: damped oscillator (x1: pos, x2: vel)
  res = Float64Tensor.zeros([2, 1])
  res[0, 0] = x[1, 0]
  res[1, 0] = -10.0 * x[0, 0] - 0.5 * x[1, 0]
  res
}

t_vec = Float64Tensor.linear_space(0.0, 10.0, 101)
x0 = [[1.0], [0.0]].to_tensor

# Vectorized solver returns {times, states} as Tensors
times, states = CrySpace::Solver.rk4(f, x0, t_vec)

# Extract trajectories
pos_trajectory = states[..., 0]
vel_trajectory = states[..., 1]

# Calculate an arbitrary output y = 2*pos + 0.1*vel using tensor math
outputs = pos_trajectory * 2.0 + vel_trajectory * 0.1

puts "Final Output: #{outputs[-1].value}"
```

### 5. Advanced Analysis (Modern Control)
CrySpace provides tools to analyze the structural properties of systems.

```crystal
# Define a system
sys = CrySpace::StateSpace.new(a, b, c, d)

# Check stability
puts "Is stable? #{sys.is_stable?}"

# Controllability and Observability matrices
ctrb_matrix = sys.ctrb
obsv_matrix = sys.obsv

# Rank checks
puts "Is controllable? #{sys.is_controllable?}"
puts "Is observable? #{sys.is_observable?}"
```

### 6. Transfer Function Arithmetic
You can combine Transfer Functions using standard operators.

```crystal
# G1(s) = 1 / (s + 1)
tf1 = CrySpace::TransferFunction.new([1.0].to_tensor, [1.0, 1.0].to_tensor)

# G2(s) = 1 / (s + 2)
tf2 = CrySpace::TransferFunction.new([1.0].to_tensor, [1.0, 2.0].to_tensor)

# Parallel: G1 + G2
tf_sum = tf1 + tf2

# Series: G1 * G2
tf_mul = tf1 * tf2

# Feedback: G1 / (1 + G1*G2)
tf_cl = tf1.feedback(tf2)

puts "Closed loop poles: #{tf_cl.poles}"
puts "Closed loop zeros: #{tf_cl.zeros}"
```

### 7. Bidirectional Conversions
Easily switch between State-Space and Transfer Function representations.

```crystal
# State-Space to Transfer Function
tf = sys.to_transferfunction

# Transfer Function to State-Space (Controllable Canonical Form)
ss = tf.to_statespace
```

## Testing

Run the specs to ensure everything is working correctly:
```bash
crystal spec
```

## Contributing

1. Fork it (<https://github.com/eltony81/cryspace/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- [eltony81](https://github.com/eltony81) - creator and maintainer
