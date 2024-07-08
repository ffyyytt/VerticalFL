import tensorflow as tf

from keras import backend as K
from .TriggerLayer import TriggerLayer
from classification_models.keras import Classifiers

def model_factory(backbone: str = "resnet18", n_classes: int = 10, n_attackers: int = 1, n_party: int = 2):
    inputs = []
    features = []
    attackerClassifiers = []
    for i in range(n_party):
        inputImage = tf.keras.layers.Input(shape = (None, None, 3), dtype=tf.float32, name = f'image_{i}')
        B, _ = Classifiers.get(backbone)
        seqB = tf.keras.Sequential([B(input_shape = (None, None, 3), weights='imagenet', include_top=False)], name=f"backbone_{i}")
        feature = tf.keras.layers.GlobalAveragePooling2D()(seqB(inputImage))

        inputs.append(inputImage)
        features.append(feature)

    fcnn4 = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')], name = "output")
    output = fcnn4(tf.keras.layers.Concatenate()(features))

    model = tf.keras.models.Model(inputs = inputs, outputs = [output])

    for i in range(n_attackers):
        attackerModel = tf.keras.models.Model(inputs = [inputs[i]], outputs = [tf.keras.layers.Dense(n_classes, activation='softmax', name="output")(features[i])])
        attackerModel = tf.keras.models.clone_model(attackerModel)
        attackerModel.layers[-3].trainable = False
        attackerClassifiers.append(attackerModel)

    return model, attackerClassifiers

def buildTriggerModel(model, windowSize, idx):
    x_sub_s = tf.keras.layers.Input(shape = (None, None, 3), dtype=tf.float32, name = f'x_sub_s')
    x_t = tf.keras.layers.Input(shape = (None, None, 3), dtype=tf.float32, name = f'x_t')
    masks = tf.keras.layers.Input(shape = (windowSize, windowSize, 3, None, None, 3), dtype=tf.float32, name = f'masks')

    x_hat_s = TriggerLayer(windowSize=windowSize)([x_sub_s, masks])

    newModel = tf.keras.models.clone_model(model)
    backbone = newModel.get_layer(f"backbone_{idx}")
    backbone.trainable = False

    f1 = backbone(x_hat_s)
    f2 = backbone(x_t)
    eq1 = tf.keras.layers.Lambda(lambda x: K.sum(K.abs(x), axis=-1, keepdims=True), name='euclidean_distance')(tf.keras.layers.subtract([f1, f2]))
    distance = tf.keras.layers.Flatten(name="output")(eq1)

    triggerModel = tf.keras.models.Model(inputs = [x_sub_s, x_t, masks], outputs = [distance])
    return triggerModel