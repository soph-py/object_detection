from flask import Flask, request, jsonify
from src.model_utils import Yolo

app = Flask(__name__)

# path = 'static/best.pt'
# yolo_model = Yolo(weights_=path)

@app.route('/', methods=['POST'])
def batch_request():
    if request.method == 'POST':
        files = request.files.getlist('files')
        post_request_data = request.form.to_dict(flat=False)['index']
        if not files:
            return
        else:
            yolo_model.files = files
            predictions = yolo_model.get_predictions() # list of predictions
        detection_map = dict(index=post_request_data, detections=predictions)
        return jsonify(detection_map)

if __name__ == '__main__':
    path = 'static/best.pt'
    yolo_model = Yolo(weights_=path)
    app.run(debug=True)