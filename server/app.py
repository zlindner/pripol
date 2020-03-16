from flask import Flask, render_template

def create_app():
    app = Flask(__name__, static_folder='../dist', template_folder='../dist')

    register_routes(app)

    return app

def register_routes(app):
    @app.route('/<path:path>')
    def serve(path):
        return render_template('index.html')