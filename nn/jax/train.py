import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import tensorflow_datasets as tfds
from BabyCNN import BabyCNN
from get_dataloaders import get_dataloaders


class TrainState(train_state.TrainState):
    pass


@jax.jit
def apply_model(state, images, labels):
    def loss_fn(params):
        logits = BabyCNN().apply({"params": params}, images)
        loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    return state, loss, accuracy


@jax.jit
def eval_model(params, images, labels):
    logits = BabyCNN().apply({"params": params}, images)
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    return loss, accuracy


def train_and_evaluate(num_epochs=10, batch_size=64, learning_rate=0.001):
    ds_train, ds_test = get_dataloaders(batch_size=batch_size)

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.ones([1, 28, 28, 1], dtype=jnp.float32)
    params = BabyCNN().init(init_rng, dummy_input)["params"]

    # create train state
    tx = optax.adam(learning_rate)
    state = TrainState.create(apply_fn=BabyCNN().apply, params=params, tx=tx)

    # training loop
    for epoch in range(num_epochs):
        # train
        for batch in tfds.as_numpy(ds_train):
            images, labels = batch
            images = images.reshape(
                (images.shape[0], 28, 28, 1)
            )  # ensure correct shape
            state, loss, accuracy = apply_model(state, images, labels)
        print(f"Epoch {epoch+1}, Loss: {loss}, Accuracy: {accuracy*100}%")

        # evaluate
        test_loss, test_accuracy = 0, 0
        num_batches = 0
        for batch in tfds.as_numpy(ds_test):
            images, labels = batch
            images = images.reshape(
                (images.shape[0], 28, 28, 1)
            )  # ensure correct shape
            loss, accuracy = eval_model(state.params, images, labels)
            test_loss += loss
            test_accuracy += accuracy
            num_batches += 1
        test_loss /= num_batches
        test_accuracy /= num_batches
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy*100}%")


if __name__ == "__main__":
    train_and_evaluate()
