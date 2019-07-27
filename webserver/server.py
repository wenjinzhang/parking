from flask import Flask
from transform import split
from checkSpaces import checkSpaces
from flask import render_template
import cv2

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


def main():
    split("./static/play2.jpg", 7, 17)
    checkSpaces("./static/play2.jpg")


if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=5000, debug=True)
    main()