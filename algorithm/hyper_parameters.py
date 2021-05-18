# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

## The following flags are related to save paths, tensorboard outputs and screen outputs

tf.app.flags.DEFINE_string('version', 'JOURNAL_RESNET_REAL_3_1', '''A version number defining the directory to save
logs and checkpoints''')
#tf.app.flags.DEFINE_integer('report_freq', 1000, '''Steps takes to output errors on the screen
#and write summaries''')
tf.app.flags.DEFINE_integer('report_freq', 1, '''Steps takes to output errors on the screen
and write summaries''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's
moving average shown on tensorboard''')


## The following flags define hyper-parameters regards training
#tf.app.flags.DEFINE_integer('sample_size', 7899, '''Total number of samples that you want to train''') 24
tf.app.flags.DEFINE_integer('sample_size', 300, '''Total number of samples that you want to train''')
#tf.app.flags.DEFINE_integer('train_steps', 50001, '''Total steps that you want to train''')500
tf.app.flags.DEFINE_integer('train_steps', 500, '''Total steps that you want to train''')
tf.app.flags.DEFINE_boolean('is_full_validation', True, '''Validation w/ full validation set or
a random batch''')
#tf.app.flags.DEFINE_integer('train_batch_size', 100, '''Train batch size''') 10
tf.app.flags.DEFINE_integer('train_batch_size', 300, '''Train batch size''')
# tf.app.flags.DEFINE_integer('validation_batch_size', 1000, '''Validation batch size, better to be
# a divisor of 10000 for this task''') 10
tf.app.flags.DEFINE_integer('validation_batch_size', 300, '''Validation batch size, better to be
# a divisor of 10000 for this task''')
tf.app.flags.DEFINE_integer('test_batch_size', 25, '''Test batch size''')

# tf.app.flags.DEFINE_float('init_lr', 0.1, '''Initial learning rate''')0.05
tf.app.flags.DEFINE_float('init_lr', 0.06, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.1, '''How much to decay the learning rate each
time''')
# tf.app.flags.DEFINE_integer('decay_step0', 40000, '''At which step to decay the learning rate''')
# tf.app.flags.DEFINE_integer('decay_step1', 60000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step0', 4, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 6, '''At which step to decay the learning rate''')


## The following flags define hyper-parameters modifying the training network

#tf.app.flags.DEFINE_integer('num_residual_blocks', 18, '''How many residual blocks do you want''')
tf.app.flags.DEFINE_integer('num_residual_blocks', 5, '''How many residual blocks do you want''')
tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')


## The following flags are related to data-augmentation

tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on
each side of the image''')


## If you want to load a checkpoint and continue training

tf.app.flags.DEFINE_string('ckpt_path', 'cache/logs_repeat20/model.ckpt-100000', '''Checkpoint
directory to restore''')
tf.app.flags.DEFINE_boolean('is_use_ckpt', False, '''Whether to load a checkpoint and continue
training''')

tf.app.flags.DEFINE_string('test_ckpt_path', 'model500_0.6_500_0.06_new.ckpt-499', '''Checkpoint
directory to restore''')


train_dir = 'logs_' + FLAGS.version + '/'
# train_dir = '/'