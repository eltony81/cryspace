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

### 1. Basic Example: First-Order System (Non-vectorized)
A simple starting point using manual iteration for a first-order system ($\dot{x} = -x + u$).

```crystal
require "cryspace"

# G(s) = 1 / (s + 1)
a = [[-1.0]].to_tensor
b = [[1.0]].to_tensor
c = [[1.0]].to_tensor
d = [[0.0]].to_tensor

sys = CrySpace::StateSpace.new(a, b, c, d)

# Non-vectorized step response (manual loop)
t_arr, x_arr, y_arr = sys.step_response(n_steps: 10)

t_arr.each_with_index do |time, i|
  state = x_arr[i][0, 0].value
  output = y_arr[i][0, 0].value
  puts "t: #{time.round(1)}s, State: #{state.round(4)}, Output: #{output.round(4)}"
end
```

### 2. Mechanical Example: Mass-Spring-Damper System
A classic 2-state matrix system representing a physical oscillator.

**The Physical Setup:**
Newton's Second Law for a mass $m$, spring stiffness $k$, and damping coefficient $c$ with an external force $u$:
```math
m\ddot{y} + c\dot{y} + ky = u
```

**Implementation (Vectorized):**
```crystal
m, k, c = 1.0, 10.0, 0.5

a = [[0.0, 1.0], [-k/m, -c/m]].to_tensor
b = [[0.0], [1/m]].to_tensor
c = [[1.0, 0.0]].to_tensor
d = [[0.0]].to_tensor

sys_msd = CrySpace::StateSpace.new(a, b, c, d)

# Analyze structural properties
puts "Is Stable? #{sys_msd.is_stable?}"
puts "Is Controllable? #{sys_msd.is_controllable?}"
puts "Is Observable? #{sys_msd.is_observable?}"

# Vectorized simulation from 0 to 10s
t = Float64Tensor.linear_space(0.0, 10.0, 101)
times, states, outputs = sys_msd.simulate(t)

# Access trajectories
pos = states[..., 0]
vel = states[..., 1]
y_out = outputs[..., 0]

puts "Max position: #{pos.max.value}"
```

### 3. Comprehensive Example: RLC Circuit with PID Control
This step-by-step guide shows how to model a physical system, design a PID controller, and simulate the closed-loop performance.

#### Step 1: Define the Physical Plant (RLC)
We model a second-order RLC circuit with Capacitor Voltage ($v_C$) and Inductor Current ($i_L$) as states.
```math
A = \begin{bmatrix} 0 & 1/C \\ -1/L & -R/L \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 1/L \end{bmatrix}
```

```crystal
require "cryspace"

# Circuit parameters
R, L, C = 1.0, 0.5, 0.1

a = [[0.0, 1/C], [-1/L, -R/L]].to_tensor
b = [[0.0], [1/L]].to_tensor
c = [[1.0, 0.0]].to_tensor # We measure Capacitor Voltage
d = [[0.0]].to_tensor

rlc_plant = CrySpace::StateSpace.new(a, b, c, d)
```

#### Step 2: Design the PID Controller
We use a filtered derivative term to ensure the controller is "proper" and can be converted to State-Space.
```math
K(s) = K_p + \frac{K_i}{s} + \frac{K_d s}{T_f s + 1}
```

```crystal
kp, ki, kd = 10.0, 5.0, 1.0
tf = 0.01 # Filter time constant

# Construct PID Transfer Function
num_pid = [(kp * tf + kd), (kp + ki * tf), ki].to_tensor
den_pid = [tf, 1.0, 0.0].to_tensor
pid_tf = CrySpace::TransferFunction.new(num_pid, den_pid)

# Convert to State-Space for interconnection
pid_controller = pid_tf.to_statespace
```

#### Step 3: Create the Closed-Loop System
We connect the PID controller in negative feedback with the RLC plant.
```crystal
# Negative feedback loop
sys_cl = rlc_plant.feedback(pid_controller)
```

#### Step 4: Analyze Stability and Structural Properties
Verify if the designed control loop is stable and if all states are observable.
```crystal
puts "Is Closed-Loop Stable? #{sys_cl.is_stable?}"
puts "Is Closed-Loop Observable? #{sys_cl.is_observable?}"
puts "Closed-Loop Poles: #{sys_cl.poles}"
```

#### Step 5: Run Vectorized Simulation (0 - 120s)
Simulate the full system transient and retrieve all state trajectories.
```crystal
# Define time range
t = Float64Tensor.linear_space(0.0, 120.0, 1201)

# Vectorized simulation
times, states, outputs = sys_cl.simulate(t)

# Retrieve trajectories
# states[..., 0..1] -> RLC states
# states[..., 2..3] -> PID internal states
voltage = outputs[..., 0] 
current = states[..., 1]

puts "Steady-state voltage: #{voltage[-1].value.round(4)} V"
```

### 4. Feedback Connection (Simple Gain)
```crystal
# Closed loop with unity gain feedback
k_gain = [[1.0]].to_tensor
sys_cl = sys.feedback(k_gain)

# Simulate closed-loop response
t_cl = Float64Tensor.linear_space(0.0, 10.0, 101)
_, states_cl, outputs_cl = sys_cl.simulate(t_cl)

# Retrieve values
cl_pos = states_cl[..., 0]
cl_out = outputs_cl[..., 0]

puts "Closed loop final position: #{cl_pos[-1].value}"
```

### 5. Step Response and State Analysis
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

### 6. General ODE Solving (RK4)
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

#### Manual Iteration (Detailed)
```crystal
# Using t_span and dt returns Arrays of Tensors for the equation dx/dt = f(x, t)
times, states = CrySpace::Solver.rk4(f, x0, {0.0, 10.0}, 0.1)

times.each_with_index do |t, i|
  pos = states[i][0, 0].value
  vel = states[i][1, 0].value
  y = 2 * pos + 0.1 * vel
  puts "t: #{t.round(2)}, y: #{y.round(4)}"
end
```

### 7. Transfer Function Arithmetic
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

# To simulate, convert to State-Space
sys_tf = tf_cl.to_statespace
t_tf = Float64Tensor.linear_space(0.0, 5.0, 51)
_, states_tf, outputs_tf = sys_tf.simulate(t_tf)

puts "TF output at 5s: #{outputs_tf[-1, 0].value}"
```

### 8. Bidirectional Conversions
Easily switch between State-Space and Transfer Function representations.

```crystal
# State-Space to Transfer Function
tf = sys.to_transferfunction

# Transfer Function to State-Space (Controllable Canonical Form)
ss = tf.to_statespace

# The converted system can be simulated normally
t_ss = Float64Tensor.linear_space(0.0, 1.0, 11)
_, x_ss, y_ss = ss.simulate(t_ss)
```

## Visualization Application

Included in this repository is a plotter application that simulates systems and automatically generates a plot using Python's `matplotlib`.

### Setup
1. Ensure you have Python installed and create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install pandas matplotlib
   ```

2. Navigate to the plotter application directory:
   ```bash
   cd /path/to/plotter_app
   ```

### Running the Simulation
To run the simulation and generate the `simulation_plot.png` image:
```bash
source /path/to/venv/bin/activate
crystal run src/plotter.cr -Dopenblas
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
