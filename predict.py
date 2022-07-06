
import cv2
import tensorflow as tf


emotions = ["Angry", "Disgusted"   ,"Fearful",  "Happy", "Neutral",  "Sad",  "Surprised"]

def prepare(filepath):
    IMG_SIZE = 48
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE, IMG_SIZE,1)

model = tf.keras.models.load_model('model.h5')

predictions = model.predict([prepare('data/test/surprised/im1.jpg')])
print(
    predictions[0][0],
    predictions[0][1],
    predictions[0][2],
    predictions[0][3],
    predictions[0][4],
    predictions[0][5],
    predictions[0][6]
)

for x in range(7):
    if predictions[0][x] == 1:
        print(emotions[x])
