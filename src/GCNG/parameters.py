from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

l2_reg = 1e-6  # Regularization rate for l2
batch_size = 32
es_patience = 50  # Patience fot early stopping
optimizer = Adam(learning_rate=1*1e-6) # -6 or -7
loss_fn = BinaryCrossentropy()
num_epoch = 20000 # 20000 or 30000

first_epoch = 300
last_epoch  = num_epoch-2000