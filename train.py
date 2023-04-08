import tensorflow as tf
import model as m
from config import *
import time
import os
from utils import DataLoader
import datetime

restore_model = args.restore

data_loader = DataLoader(args.batch_size, args.T, args.data_scale,
                         chars=args.chars, points_per_char=args.points_per_char)

args.U = data_loader.max_U
args.c_dimension = len(data_loader.chars) + 1

model = m.Model()

# init model
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
loss_fn = m.compute_custom_loss

# define checkpoints and restore model and optimizer from
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(
    checkpoint, 
    directory=f"{args.tgtdir}",
    max_to_keep=1000
)
status = checkpoint.restore(manager.latest_checkpoint)

for epoch in range(200):
    data_loader.reset_batch_pointer()
    for b in range(data_loader.num_batches):
        tic = time.time()
        with tf.GradientTape() as tape:

            x, y, c_vec, c = data_loader.next_batch()
            out = model([x, c_vec])
            loss_value = loss_fn(y, out)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if b % 1 == 0:
            print(f"epoch: {epoch} - batch: {b+1}/{data_loader.num_batches} - loss: {loss_value} - time: {str(round(time.time() - tic, 2))}")
            line = f"{datetime.datetime.now(datetime.timezone.utc).isoformat()},{epoch},{b+1}/{data_loader.num_batches},{loss_value},{str(round(time.time() - tic, 2))}\n"
            with open(f"{args.tgtdir}/loss.txt", "a") as myfile:
                myfile.write(line)

    manager.save()

