import tensorflow as tf
import model as m
from config import *
import time

from utils import DataLoader

restore_model = args.restore

data_loader = DataLoader(args.batch_size, args.T, args.data_scale,
                         chars=args.chars, points_per_char=args.points_per_char)

args.U = data_loader.max_U
args.c_dimension = len(data_loader.chars) + 1

model = m.Model()
if restore_model:
    try:
        model.load_weights(args.restore)
    except():
        print("Couldn't find checkpoint-file. Continuing with empty model")

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
loss_fn = m.compute_custom_loss
for e in range(args.num_epochs):
    print("epoch %d" % e)
    data_loader.reset_batch_pointer()
    for b in range(data_loader.num_batches):
        tic = time.time()
        with tf.GradientTape() as tape:

            x, y, c_vec, c = data_loader.next_batch()
            
            out = model([x, c_vec])
            loss_value = loss_fn(y, out)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        if b % 100 == 0:
            print('batches %d/%d, loss %g -- time: %s' % (b, data_loader.num_batches, loss_value,
                                                          str(round(time.time() - tic, 2))))

    model.save_weights('lstm_validator_%d/checkpoint' % e)
