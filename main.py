import numpy as np
import tensorflow as tf
from Dual_stage_attention_model import DARNN
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc

T = 5
m = 32
p = 32
train_num = 78186 - 10000
batch_size = 512
enc_Data = np.load("data/encoder_data.npy")
dec_Data = np.load("data/decoder_data.npy")
target = np.load("data/target.npy")
model = DARNN(T=T, m=m, p=p,)

train_ds = (
    tf.data.Dataset.from_tensor_slices(
        (enc_Data[:train_num], dec_Data[:train_num], target[:train_num])
    )
    .batch(batch_size)
    .shuffle(buffer_size=train_num)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
test_ds = tf.data.Dataset.from_tensor_slices(
    (enc_Data[train_num:], dec_Data[train_num:], target[train_num:])
).batch(batch_size)


@tf.function
def train_step(model, inputs, labels, loss_fn, optimizer, train_loss):
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=True)
        loss = loss_fn(labels, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


@tf.function
def test_step(model, inputs, labels, loss_fn, test_loss):
    prediction = model(inputs, training=False)
    loss = loss_fn(labels, prediction)
    test_loss(loss)
    return prediction


loss_fn = tf.keras.losses.MSE

optimizer = tf.keras.optimizers.Adam(0.001)
train_loss = tf.keras.metrics.Mean(name="train_loss")
test_loss = tf.keras.metrics.Mean(name="test_loss")
train_accuracy = tf.keras.metrics.Accuracy(name="train_accuracy")
test_accuracy = tf.keras.metrics.Accuracy(name="test_accuracy")
Epochs = 100
for epoch in range(Epochs):
    for enc_data, dec_data, labels in train_ds:
        inputs = [enc_data, dec_data]
        train_step(model, inputs, labels, loss_fn, optimizer, train_loss)

    template = "Epoch {}, Loss: {}"
    print(template.format(epoch + 1, train_loss.result()))
    train_loss.reset_states()
    test_loss.reset_states()
#%%
i = 0
for enc_data, dec_data, label in test_ds:
    inputs = [enc_data, dec_data]
    pred = test_step(model, inputs, label, loss_fn, test_loss)
    if i == 0:
        preds = pred.numpy()
        labels = label.numpy()
        i += 1
    else:
        preds = np.concatenate([preds, pred.numpy()], axis=0)
        labels = np.concatenate([labels, label.numpy()], axis=0)
print(test_loss.result(), test_accuracy.result() * 100)

#%%
preds = np.array(preds)
labels = np.array(labels)
plt.scatter(preds, labels, color="orange")
plt.xlabel("pred")
plt.ylabel("label")
plt.show()

enc_data, dec_data, label = next(iter(test_ds))
inputs = [enc_data, dec_data]

pred = model(inputs)
beta = []
for i in range(5):
    beta.append(np.mean(model.decoder.beta_t[:, i, 0].numpy()))
plt.bar(x=range(5), height=beta, color="orange")
plt.style.use("seaborn-pastel")
plt.title("Beta")
plt.xlabel("time")
plt.ylabel("prob")
plt.show()
#%%
variable_dict = {
    "기온": ["0", "7", "28", "31", "32"],
    "현지기압": ["1", "6", "22", "27", "29"],
    "풍속": ["2", "3", "18", "24", "26"],
    "일일 누적강수량": ["4", "10", "21", "36", "39"],
    "해면기압": ["5", "8", "9", "23", "33"],
    "일일 누적일사량": ["11", "14", "16", "19", "34"],
    "습도": ["12", "20", "30", "37", "38"],
    "풍향": ["13", "15", "17", "25", "35"],
}
font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc("font", family=font_name)
variable_key = list(variable_dict.keys())
alpha = []
variables = []
for i in range(39):
    alpha.append(np.mean(model.encoder.alpha_t[:, 0, i].numpy()))
    for key in variable_key:
        if f"{i}" in variable_dict[key]:
            variables.append(f"{key}{i}")
plt.figure(figsize=(12, 8))
plt.bar(x=variables, height=alpha, color="orange")
plt.style.use("seaborn-pastel")
plt.title("alpha")
plt.xlabel("variables")
plt.xticks(rotation=90)
plt.ylabel("prob")
plt.show()
