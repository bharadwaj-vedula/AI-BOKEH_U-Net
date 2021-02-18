import tensorflow as tf
from tensorflow.keras.models import Model,load_model,save_model
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""unet= load_model("best_unet.h5")

#print(unet.summary())

converter= tf.lite.TFLiteConverter.from_keras_model(unet)
tflite_model = converter.convert()
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#converter.target_spec.supported_types = [tf.float16]
tflite_quant_model= converter.convert()
open("unet_fp16.tflite","wb").write(tflite_quant_model)"""

#testing
interpreter = tf.lite.Interpreter(model_path="unet_fp16.tflite")
interpreter.allocate_tensors()

interpreter_fp16 = tf.lite.Interpreter(model_path="unet_fp16.tflite")
interpreter_fp16.allocate_tensors()

test_image_path="elon1.jpg"
test_image=plt.imread(test_image_path)
test_image=cv2.resize(test_image,(256,256))
test_image = test_image/255
test_image = np.expand_dims(test_image, axis=0).astype(np.float32)

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

interpreter.set_tensor(input_index, test_image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)

predictions=np.resize(predictions,(256,256))

cv2.imshow("image",predictions)

