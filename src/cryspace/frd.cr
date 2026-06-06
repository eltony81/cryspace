require "num"

module CrySpace
  class FRD
    property omega : Float64Tensor
    property response : Tensor(Complex, CPU(Complex))

    def initialize(@omega : Float64Tensor, @response : Tensor(Complex, CPU(Complex)))
    end
  end
end
