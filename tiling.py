import tensorflow as tf

variables = tf.ones([3, 2, 2, 10])


print(variables.shape)
flat_variables = tf.reshape(variables, [-1, 2*2*10])
print(flat_variables.shape)

heat_map = tf.constant([[[1.0, 2.0],
                            [3.0, 4.0]],
                            [[5.0, 6.0],
                            [7.0, 8.0]],
                            [[9.0, 10.0],
                            [11.0, 12.0]]])
print(heat_map.shape)

flat_heat_map = tf.reshape(heat_map, [-1, 2*2])
heat_tile = tf.tile(flat_heat_map, [1, 10])

print(flat_heat_map.shape)
print(heat_tile.shape)
weighted_var = tf.multiply(flat_variables, heat_tile)

ans = tf.Session().run(flat_variables)
print("Flat variables: ", ans)
print("heat map tiled: ", tf.Session().run(heat_tile))
print("weighted variables: ", tf.Session().run(weighted_var))
