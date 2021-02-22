import tensorflow as tf
from preprocessing import PreProcessing
from model import TripletLoss

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 512, 'Batch size')
flags.DEFINE_integer('train_iter', 2000, 'Total training iter')
flags.DEFINE_integer('step', 50, 'Save after ... iteration')
flags.DEFINE_float('learning_rate',0.01,'Learning rate') # '0.01'?
flags.DEFINE_float('momentum',0.99, 'Momentum')
flags.DEFINE_string('model', 'conv_net', 'Model to run')
flags.DEFINE_string('data_src', '/retail_corpus', 'Source of training dataset')

if __name__ == "__main__":

    # Setup Dataset
    dataset = PreProcessing(FLAGS.data_src)
    model = TripletLoss()
    placeholder_shape = [None] + list(dataset.images_train.shape[1:]) # num rows we don't know yet + other dims
    print("placeholder_shape", placeholder_shape)

    # Setup Network
    next_batch = dataset.get_triplets_batch
    anchor_input = tf.compat.v1.placeholder(tf.float32, placeholder_shape, name='anchor_input')
    positive_input = tf.compat.v1.placeholder(tf.float32, placeholder_shape, name='positive_input')
    negative_input = tf.compat.v1.placeholder(tf.float32, placeholder_shape, name='negative_input')

    margin = 0.5
    anchor_output = model.conv_net(anchor_input, reuse=False)
    positive_output = model.conv_net(positive_input, reuse=True)
    negative_output = model.conv_net(negative_input, reuse=True)
    loss = model.triplet_loss(anchor_output, positive_output, negative_output, margin)

    # Setup Optimizer
    global_step = tf.Variable(0, trainable=False) 
    # trainable default=true i.e. GradientTapes automatically watch uses of this variable

    train_step = tf.compat.v1.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum, use_nesterov=True).minimize(loss,
                                                                                                             global_step=global_step)

    # Start Training
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess: # class to run tf operations
        sess.run(tf.compat.v1.global_variables_initializer())

        # Setup Tensorboard
        # log metrics like loss during training/testing within the scope of the summary writers
        tf.summary.scalar('step', global_step) 
        tf.summary.scalar('loss', loss) 
        for var in tf.compat.v1.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        merged = tf.compat.v1.summary.merge_all()
        writer = tf.compat.v1.summary.FileWriter('train.log', sess.graph)

        # Train iter
        for i in range(FLAGS.train_iter):
            batch_anchor, batch_positive, batch_negative = next_batch(FLAGS.batch_size)

            _, l, summary_str = sess.run([train_step, loss, merged], 
                                feed_dict={anchor_input: batch_anchor, positive_input: batch_positive, negative_input: batch_negative})

            writer.add_summary(summary_str, i)
            print("\r#%d - Loss" % i, l) #\r line break, %d is a placeholder for numeric/decimal values

            if (i + 1) % FLAGS.step == 0: # remainder is 0 left
                saver.save(sess, "trained_model/model_triplet/model.ckpt")
        saver.save(sess, "trained_model/model_triplet/model.ckpt")
    print('Training completed successfully.')