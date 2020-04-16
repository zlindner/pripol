from flask import Flask, render_template
from blueprints import policy, model

def create_app():
    app = Flask(__name__, template_folder='../dist', static_folder='../dist/js')

    register_blueprints(app)
    register_routes(app)

    return app

def register_routes(app):
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        return render_template('index.html')

def register_blueprints(app):
    app.register_blueprint(policy.bp)
    app.register_blueprint(model.bp)

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)