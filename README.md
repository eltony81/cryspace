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

### 2. Feedback Connection
```crystal
# Closed loop with unity gain feedback
k_gain = [[1.0]].to_tensor
sys_cl = sys.feedback(k_gain)
puts "Closed loop poles: #{sys_cl.poles}"
```

### 3. Step Response and State Analysis
You can simulate the system's response to a step input and obtain the trajectory of all internal states ($x$) and outputs ($y$).

```crystal
# Simulation for 5 seconds (50 steps of 0.1s)
t, x, y = sys.step_response(n_steps: 50)

puts "Time (s) | Position (Output) | Velocity (State x2)"
puts "-" * 55
t.each_with_index do |time, i|
  # Accessing internal states (x)
  # x[i][0, 0] is state 1 (position)
  # x[i][1, 0] is state 2 (velocity)
  x1 = x[i][0, 0].value
  x2 = x[i][1, 0].value
  
  # Accessing system output (y)
  # y[i][0, 0] is the output (measuring position)
  output = y[i][0, 0].value
  
  puts "#{time.round(2).to_s.ljust(8)} | #{output.round(4).to_s.ljust(17)} | #{x2.round(4)}"
end
```

### 4. General ODE Solving (RK4)
You can solve arbitrary ODEs ($\dot{x} = f(x, t)$) and calculate derived outputs:
```crystal
f = ->(x : Float64Tensor, t : Float64) {
  # Example: damped oscillator (x1: pos, x2: vel)
  res = Float64Tensor.zeros([2, 1])
  res[0, 0] = x[1, 0]
  res[1, 0] = -10.0 * x[0, 0] - 0.5 * x[1, 0]
  res
}

x0 = [[1.0], [0.0]].to_tensor
times, states = CrySpace::Solver.rk4(f, x0, {0.0, 10.0}, 0.1)

# Calculate an arbitrary output y = 2*pos + 0.1*vel
times.each_with_index do |t, i|
  pos = states[i][0, 0].value
  vel = states[i][1, 0].value
  y = 2 * pos + 0.1 * vel
  puts "t: #{t.round(2)}, y: #{y.round(4)}"
end
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
