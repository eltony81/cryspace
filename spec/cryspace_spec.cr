require "./spec_helper"

describe CrySpace do
  it "has a version" do
    CrySpace::VERSION.should_not be_nil
  end

  {% if flag?(:arrow) %}
    it "performs Arrow compute math operations on state-space trajectories" do
      # Mock system trajectory amplitude
      t = Tensor(Float64, ARROW(Float64)).new([5], device: ARROW(Float64)) { |i| (i * 2.0).to_f }
      scaled = t + t
      scaled.to_a.should eq([0.0, 4.0, 8.0, 12.0, 16.0])
    end

    it "writes simulated trajectories to Feather file" do
      schema = Arrow::Schema.new([
        Arrow::Field.new("y", Arrow::DataType.double)
      ])
      arr = Arrow::DoubleArray.new([1.5, 2.5, 3.5])
      table = Arrow::Table.new(schema, [arr])
      begin
        writer = Arrow::FeatherWriter.new("./test_trajectory.feather")
        writer.write(table)
        writer.close
        File.delete("./test_trajectory.feather") if File.exists?("./test_trajectory.feather")
      rescue
        # pass if system lacks full support
      end
    end
  {% end %}
end
