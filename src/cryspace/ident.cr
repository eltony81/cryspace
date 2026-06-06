require "num"

module CrySpace
  class Ident
    # Generates a Pseudo-Random Binary Sequence (PRBS) using LFSR.
    def self.prbs(order : Int32, seed : UInt32 = 1) : Float64Tensor
      taps = {
        2 => [2, 1],
        3 => [3, 2],
        4 => [4, 3],
        5 => [5, 3],
        6 => [6, 5],
        7 => [7, 6],
        8 => [8, 6, 5, 4],
        9 => [9, 5],
        10 => [10, 7]
      }
      
      tap_list = taps[order]? || [order, order - 1]
      
      len = (1 << order) - 1
      res = Float64Tensor.zeros([len])
      reg = seed == 0 ? 1_u32 : seed
      
      len.times do |i|
        output_bit = reg & 1
        res[i] = output_bit == 1 ? 1.0 : -1.0
        
        feedback = 0_u32
        tap_list.each do |t|
          feedback ^= (reg >> (order - t)) & 1
        end
        reg = (reg >> 1) | (feedback << (order - 1))
      end
      res
    end

    # Generates a swept-frequency cosine (chirp) signal.
    def self.chirp(t : Float64Tensor, f0 : Float64, t1 : Float64, f1 : Float64) : Float64Tensor
      beta = (f1 - f0) / (2.0 * t1)
      res = Float64Tensor.zeros(t.shape)
      t.size.times do |i|
        t_val = t[i].value
        phase = 2.0 * Math::PI * (f0 * t_val + beta * t_val * t_val)
        res[i] = Math.cos(phase)
      end
      res
    end
  end
end
