from ultralytics import YOLO
from flask import request, Response, Flask
from waitress import serve
from PIL import Image
import json

app = Flask(__name__)

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("D:\\Python files\\yolov8\\templates\\index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file", 
    passes it through YOLOv8 object detection 
    network and returns an array of bounding boxes.
    :return: a JSON array of objects bounding 
    boxes in format 
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    return Response(
        json.dumps(boxes),  
        mimetype='application/json'
    )


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format 
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = YOLO("D:/Python files/yolov8/best (1).pt")
    results = model.predict(buf)
    
    output = []  # Initialize the output list
    for result in results:
        for box in result.boxes:
            for i in range(len(box.xyxy)):
                  x1, y1, x2, y2 = [int(x.item()) for x in box.xyxy[i]]
                  class_id = box.cls[i].item()
                  prob = round(box.conf[i].item(), 2)
                  output.append([x1, y1, x2, y2, result.names[class_id], prob])
    
    return output


if __name__ == '__main__':
    app.run(debug=True)
