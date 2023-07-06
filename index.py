from flask import Flask, render_template, request, jsonify
import subprocess
import json
import os
import ast
import time
from datetime import datetime, timedelta
from QueryProcessing import get_relevent_score
from TextClassification import naive_bayes_classification
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("search.html")


@app.route("/run_crawler", methods=["POST"])
def run_crawler():
    filename = "schedule.json"
    # open file if exist
    if os.path.exists(filename):
        with open(filename, "r") as file:
            data = json.load(file)

        # Extract the date from the JSON data
        json_date = datetime.strptime(data["last_scan"], "%Y-%m-%d %H:%M:%S")

        # Compare the dates
        one_week_ago = datetime.now() - timedelta(weeks=1)
        if json_date < one_week_ago:

            #update scan time at first so that another scan doesn't happen at same time
            data["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(filename, "w") as file:
                json.dump(data, file)

            #for calculating execution time
            start_time = time.time()

            # run crawler, wait and run data_strucutres for indexing
            crawler = subprocess.Popen(["python", "crawler.py"])
            crawler.wait()

            InvertedIndex = subprocess.Popen(["python", "InvertedIndex.py"])
            InvertedIndex.wait()

            end_time = time.time()

            execution_time = end_time - start_time
            execution_time_formatted = "{:.2f}".format(execution_time)

            
            return jsonify(
                {"message": "Data successfully scrapped and index updated in "+execution_time_formatted+"s"}
            )
        else:
            return jsonify({"message": "Update time limit not reached."})

    else:
        # write and scan if file doesn't exist
        data = {"last_scan": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        with open(filename, "w") as file:
            json.dump(data, file)

        start_time = time.time()
        # run crawler, wait and run data_strucutres for indexing
        crawler = subprocess.Popen(["python", "crawler.py"])
        crawler.wait()

        InvertedIndex = subprocess.Popen(["python", "InvertedIndex.py"])
        InvertedIndex.wait()

        end_time = time.time()

        execution_time = end_time - start_time
        execution_time_formatted = "{:.2f}".format(execution_time)

        return jsonify({"message": "Data successfully scrapped and index updated in "+execution_time_formatted+"s"})

    return jsonify({"message": "Data is upto date."})


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query")
    page = int(request.args.get("page", 1))
    per_page = 5  # Number of results to display per page

    # Perform search operation with the query
    start_time = time.time()
    results = get_relevent_score(query)

    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_formatted = "{:.2f}".format(execution_time)


    total_results = len(results)
    num_pages = (total_results + per_page - 1) // per_page
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated_results = results[start_index:end_index]

    # evaluate data for rendering in jinja2
    for item in paginated_results:
        item["RCIH_authors"] = ast.literal_eval(item["RCIH_authors"])
        item["authors"] = ast.literal_eval(item["authors"])

    return render_template(
        "search.html",
        results=paginated_results,
        total_results=total_results,
        query=query,
        page=page,
        prev_page=page - 1,
        next_page=page + 1 if page < num_pages else None,
        page_nums=range(1, num_pages + 1),
        current_page=page,
        time = execution_time_formatted
    )


@app.route("/text_classifier", methods=["GET", "POST"])
def text_classifier_method():
    if request.method == "POST":
        text = request.form.get("text")
        
        start_time = time.time()
        results = naive_bayes_classification(text)

        end_time = time.time()
        execution_time = end_time - start_time
        execution_time_formatted = "{:.2f}".format(execution_time)
        
        return render_template("text_classifier.html", text=text, results=results, time = execution_time_formatted)

    return render_template("text_classifier.html")


if __name__ == "__main__":
    app.run()
