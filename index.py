from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    # Perform search operation with the query
    # Replace the following line with your search logic
    results = ['Result 1', 'Result 2', 'Result 3']
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run()