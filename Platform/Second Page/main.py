import cv2
import numpy as np
import requests
import io
import json
from glob import glob
pth ="E:\xampp\htdocs\Squadfree\imagesuploadedf"

img = cv2.imread(glob(pth+"*.jpg"))
height, width , _ = img.shape
#print(img.shape)
img = img[0: height, 400:width]
#OCR
url_api = "https://api.ocr.space/parse/image"
_, compressedimage = cv2.imencode(".jpg", img, [1, 90])
file_bytes = io.BytesIO(compressedimage)

result = requests.post(url_api, 
              files = {"roi-300x213.jpg": file_bytes},
              data = {"apikey": "54cb02292688957"})

result = result.content.decode()
result = json.loads(result)
text_detected = result.get("ParsedResults")[0].get("ParsedText")
m = {'result': text_detected}
n = json.dumps(m)
print(n)
"""
cv2.imshow("Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""