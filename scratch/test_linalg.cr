require "../src/cryspace"

m = [[1.0, 0.5], [0.5, 1.0]].to_tensor
res = m.svd
puts "SVD return class: #{res.class}"
puts "SVD return value: #{res}"
