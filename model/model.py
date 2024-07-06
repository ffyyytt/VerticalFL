import copy
import keras_cv
import tensorflow as tf

def model_factory(backbone: str = "resnet50_imagenet", n_classes: int = 10, n_attackers: int = 1, n_party: int = 2):
    inputs = []
    features = []
    attackerClassifiers = []
    for i in range(n_party):
        inputImage = tf.keras.layers.Input(shape = (None, None, 3), dtype=tf.uint8, name = f'image_{i}')
        image = tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="tf"))(inputImage)
        feature = tf.keras.layers.GlobalAveragePooling2D()(keras_cv.models.ResNetBackbone.from_preset(backbone, name = f"feature_{i}")(image))

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
        attackerClassifiers.append(copy.deepcopy(attackerModel))
    return model, attackerClassifiers