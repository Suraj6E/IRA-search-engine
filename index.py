from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_crawler', methods=['POST'])
def run_crawler():
    subprocess.call(['python', 'crawler.py'])
    return jsonify({'message': 'Crawler executed successfully'})

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    page = int(request.args.get('page', 1))
    per_page = 5  # Number of results to display per page

    # Perform search operation with the query
    results = [
        {
            'title': 'Sample Title 1',
            'author': 'John Doe',
            'date_created': '2022-01-01',
            'date_updated': '2022-02-01'
        },
        {
            'title': 'Sample Title 2',
            'author': 'Jane Smith',
            'date_created': '2022-03-01',
            'date_updated': '2022-04-01'
        },
        {
            'title': 'Sample Title 3',
            'author': 'Bob Johnson',
            'date_created': '2022-05-01',
            'date_updated': '2022-06-01'
        },
        {
            'title': 'Sample Title 3',
            'author': 'Bob Johnson',
            'date_created': '2022-05-01',
            'date_updated': '2022-06-01'
        },
        {
            'title': 'Sample Title 3',
            'author': 'Bob Johnson',
            'date_created': '2022-05-01',
            'date_updated': '2022-06-01'
        },
        {
            'title': 'Sample Title 3',
            'author': 'Bob Johnson',
            'date_created': '2022-05-01',
            'date_updated': '2022-06-01'
        },
        {
            'title': 'Sample Title 3',
            'author': 'Bob Johnson',
            'date_created': '2022-05-01',
            'date_updated': '2022-06-01'
        },
        {
            'title': 'Sample Title 3',
            'author': 'Bob Johnson',
            'date_created': '2022-05-01',
            'date_updated': '2022-06-01'
        },
        {
            'title': 'Sample Title 3',
            'author': 'Bob Johnson',
            'date_created': '2022-05-01',
            'date_updated': '2022-06-01'
        }
    ]
    total_results = len(results)
    num_pages = (total_results + per_page - 1) // per_page
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated_results = results[start_index:end_index]

    return render_template('index.html', results=paginated_results, query=query, page=page, prev_page=page - 1,
                           next_page=page + 1 if page < num_pages else None,
                           page_nums=range(1, num_pages + 1), current_page=page)

if __name__ == '__main__':
    app.run()