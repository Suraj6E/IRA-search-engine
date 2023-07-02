from flask import Flask, render_template, request, jsonify
import subprocess
import json
import os
from datetime import datetime, timedelta

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_crawler', methods=['POST'])
def run_crawler():
    
    filename = "schedule.json"
    #open file if exist
    if os.path.exists(filename):
        with open(filename, "r") as file:
            data = json.load(file);

        # Extract the date from the JSON data
        json_date = datetime.strptime(data["last_scan"], "%Y-%m-%d %H:%M:%S")
        
        # Compare the dates
        one_week_ago = datetime.now() - timedelta(weeks=1)
        if json_date < one_week_ago:

            #run crawler, wait and run data_strucutres for indexing
            crawler = subprocess.Popen(['python', 'crawler.py'])
            crawler.wait();
            
            data_structures = subprocess.Popen(['python', 'data_structures.py'])
            data_structures.wait();

            data["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(filename, "w") as file:
                json.dump(data, file);
            
            return jsonify({'message': 'Data successfully scrapped and index updated. '})
        else:
            return jsonify({'message': "Update time limit not reached."});

    else:
        #write and scan if file doesn't exist
        data={
            "last_scan": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(filename, "w") as file:
            json.dump(data, file)

        #run crawler, wait and run data_strucutres for indexing
        crawler = subprocess.Popen(['python', 'crawler.py'])
        crawler.wait();
        
        data_structures = subprocess.Popen(['python', 'data_structures.py'])
        data_structures.wait();

        return jsonify({'message': 'Data successfully scrapped and index updated. '})

    return jsonify({'message': 'Data is upto date.'})

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