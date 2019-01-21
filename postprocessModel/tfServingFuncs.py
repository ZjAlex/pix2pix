import subprocess
import tensorflow as tf


def freeze_graph(model_folder):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    export_path = 'D:/pythonprojects/pix2pix/trainedModel/00001'


    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # op = graph.get_operation_by_name('image_tensor').outputs[0]
        # print(op)

        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input': graph.get_operation_by_name('image_tensor').outputs[0]},
            outputs={'output': graph.get_operation_by_name('generate_output/output').outputs[0]})


#freeze_graph('D:/pythonprojects/pix2pix/trainedModel')
export_path = 'D:/pythonprojects/pix2pix/trainedModel/00001'
subprocess.getstatusoutput('saved_model_cli show --dir {export_path} --all')