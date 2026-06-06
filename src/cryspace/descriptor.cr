require "num"

module CrySpace
  class DescriptorStateSpace
    property e : Float64Tensor
    property a : Float64Tensor
    property b : Float64Tensor
    property c : Float64Tensor
    property d : Float64Tensor
    property dt : Float64?

    def initialize(@e : Float64Tensor, @a : Float64Tensor, @b : Float64Tensor, @c : Float64Tensor, @d : Float64Tensor, @dt : Float64? = nil)
    end
  end
end
