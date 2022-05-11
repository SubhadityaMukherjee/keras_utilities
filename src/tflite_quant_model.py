import tensorflow_model_optimization as tfmot
import tensorflow as tf
import tfjs

model = tfmot.sparsity.keras.prune_low_magnitude(model)
tfjs.converters.save_keras_model(model, "./web/effnet")
# %%
converter = tf.lite.TFLiteConverter.from_saved_model("./web/effnet")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
# %%
with open("./web/working_quantized.tflite", "wb") as output_file:
    output_file.write(tflite_quant_model)
