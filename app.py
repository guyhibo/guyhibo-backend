from flask import request, Flask
from translate import translate_video
import json


app = Flask(__name__)
app.debug = True

@app.route('/', methods=['GET'])
def main_page():
    return "ONLINE"

@app.route('/translate', methods=['POST'])
def translate():
    params = json.loads(request.get_data())
    if len(params) == 0:
        return 'No params'
    
    filename = params["filename"]
    sentence = translate_video(filename)
    return sentence

if __name__ == "__main__":
    app.run(port=8900)