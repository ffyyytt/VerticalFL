import tensorflow as tf
from classification_models.keras import Classifiers

def model_factory(backbone: str = "resnet18", n_classes: int = 10, n_attackers: int = 1, n_party: int = 2):
    inputs = []
    features = []
    attackerClassifiers = []
    for i in range(n_party):
        inputImage = tf.keras.layers.Input(shape = (None, None, 3), dtype=tf.float32, name = f'image_{i}')
        B, _ = Classifiers.get(backbone)
        feature = tf.keras.layers.GlobalAveragePooling2D()(B(input_shape = (None, None, 3), weights='imagenet', include_top=False, name = f"feature_{i}")(inputImage))

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
        attackerClassifiers.append(tf.keras.models.clone_model(attackerModel))
    return model, attackerClassifiers