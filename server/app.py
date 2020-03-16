from flask import Flask, render_template, request

# TODO start using blueprints

def create_app():
    app = Flask(__name__, template_folder='../dist', static_folder='../dist/js')

    register_routes(app)

    return app

def register_routes(app):
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        return render_template('index.html')

    @app.route('/load-policy', methods=['POST'])
    def load_policy():
        url = request.get_json()['url']

        print(url)

        return '', 200

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)