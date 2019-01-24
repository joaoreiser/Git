import tensorflow as tf

#example save files generated from TensorFlow
'''
network.meta
network.index
network.data-00000-of-00001
'''
#WRITE your filename here
file = './network'

#WRITE your output node of the network
output_node = "fully_connected/output/Softmax"

#to find out the output node of the network, run the command below at the end of
#the network file, when creating the save files, and look for the final node
'''
for v in sess.graph.get_operations():
    print(v.name)
'''

#save files from TensorFlow
modelo_meta = file + '.meta'
modelo = file
#Generated Frozen File 
output_graph = file + '.pb'  


saver = tf.train.import_meta_graph(modelo_meta, clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, modelo)

output_node_names = output_node
output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                             input_graph_def,
                                                             output_node_names.split(",")
                                                             )
with tf.gfile.GFile(output_graph, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
    
sess.close()
