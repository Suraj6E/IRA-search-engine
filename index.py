from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    # Perform search operation with the query
    # You can add your search logic here
    return f"Search results for: {query}"

if __name__ == '__main__':
    app.run()