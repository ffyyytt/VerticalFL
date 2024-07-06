import keras_cv
import tensorflow as tf

def model_factory(backbone: str = "resnet18", n_classes: int = 10, n_attackers: int = 1, n_party: int = 2):
    inputs = []
    features = []
    backbones = []
    attackerClassifiers = []
    for i in range(n_party):
        inputImage = tf.keras.layers.Input(shape = (None, None, 3), dtype=tf.uint8, name = f'image_{i}')
        image = tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="tf"))(inputImage)
        backbones.append(keras_cv.models.ResNetBackbone.from_preset(backbone, name = f"feature_{i}"))
        feature = tf.keras.layers.GlobalAveragePooling2D()(backbones[-1](image))

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
        image = tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="tf"))(inputs[i])
        feature = tf.keras.layers.GlobalAveragePooling2D()(tf.keras.models.clone_model(backbones[i])(image))

        fcnn4 = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(n_classes, activation='softmax')], name = "output")
        output = fcnn4(tf.keras.layers.Concatenate()(features))
        
        attackerModel = tf.keras.models.Model(inputs = [inputs[i]], outputs = [output])
        attackerClassifiers.append(attackerModel)
    return model, attackerClassifiers