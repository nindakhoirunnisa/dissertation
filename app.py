from flask import Flask, render_template, request, send_file, json, jsonify
from pymongo import MongoClient
import pandas as pd
import numpy as np
import requests

app = Flask(__name__)

# MongoDB configuration
client = MongoClient('mongodb+srv://nindakhrnns:Mongodb0510@amazone.7o8cfqw.mongodb.net/?retryWrites=true&w=majority')
db = client['news']
collection = db['news']

latest_date_doc = collection.find_one(
        {},
        {'date_published': 1},
        sort=[('date_published', -1)]
    )

if latest_date_doc:
    latest_date = latest_date_doc.get('date_published')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_date_range')
def get_date_range():
    latest_date_doc = collection.find_one(
        {},
        {'date_published': 1},
        sort=[('date_published', -1)]
    )
    max_date = latest_date_doc.get('date_published')

    earliest_date_doc = collection.find_one(
        {},
        {'date_published': 1},
        sort=[('date_published', 1)]
    )
    min_date = earliest_date_doc.get('date_published')
    return {'min': min_date, 'max': max_date}

@app.route('/get_max_count')
def get_max_count():
    pipeline = [
        {
            '$unwind': {
                'path': '$all_provinces'
            }
        }, {
            '$match': {
                'all_provinces': {
                    '$ne': ''
                }
            }
        }, {
            '$group': {
                '_id': {
                    'date_published': '$date_published', 
                    'province': '$all_provinces'
                }, 
                'count': {
                    '$sum': 1
                }
            }
        }, {
            '$project': {
                '_id': 0, 
                'date_published': '$_id.date_published', 
                'province': '$_id.province', 
                'count': 1
            }
        }, {
            '$sort': {
                'count': -1
            }
        }, {
            '$limit': 1
        }, {
            '$project': {
                'max': '$count'
            }
        }
    ]

    result = list(collection.aggregate(pipeline))
    return {'max': result[0]['max']}

@app.route('/get_geojson')
def get_geojson():
    # with open('C:/Users/acer/OneDrive - The University of Manchester/Coding-dissertation/dissertation/web/geojson/indonesia-province.json', 'r') as geojson_file:
    #     data = json.load(geojson_file)
    # return json.dumps(data)
    # geojson = requests.get(
    # "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province-simple.json").json()
    # return geojson
    return send_file('C:/Users/acer/OneDrive - The University of Manchester/Coding-dissertation/dissertation/web/geojson/all_maps_state_indo.geojson', mimetype='application/json')
    
@app.route('/get_news_count', methods=['GET'])
def get_negative_news_t():
    credibility = int(request.args.get('credibility', default=2))
    date = request.args.get('date', default=latest_date)
    pipeline = [
        {
            '$match': {
                'date_published': date, 
                'credibility': credibility
            }
        }, {
            '$unwind': {
                'path': '$all_provinces'
            }
        }, {
            '$match': {
                'all_provinces': {
                    '$ne': ''
                }
            }
        }, {
            '$group': {
                '_id': '$all_provinces', 
                'count': {
                    '$sum': 1
                }, 
                'positiveCount': {
                    '$sum': {
                        '$cond': [
                            {
                                '$eq': [
                                    '$sentiment', 1
                                ]
                            }, 1, 0
                        ]
                    }
                }, 
                'negativeCount': {
                    '$sum': {
                        '$cond': [
                            {
                                '$eq': [
                                    '$sentiment', -1
                                ]
                            }, 1, 0
                        ]
                    }
                }, 
                'neutralCount': {
                    '$sum': {
                        '$cond': [
                            {
                                '$eq': [
                                    '$sentiment', 0
                                ]
                            }, 1, 0
                        ]
                    }
                }
            }
        }
    ]

    result = list(collection.aggregate(pipeline))
    return jsonify(result)

@app.route('/get_most_negative_provs', methods=['GET'])
def get_most_negatice_provs():
    pipeline = [
        {
            '$match': {
                'sentiment': -1
            }
        }, {
            '$unwind': {
                'path': '$all_provinces'
            }
        }, {
            '$group': {
                '_id': '$all_provinces', 
                'total': {
                    '$sum': 1
                }
            }
        }, {
            '$match': {
                '_id': {
                    '$ne': '甘肃省'
                }
            }
        }, {
            '$match': {
                '_id': {
                    '$nin': [
                        'Madhya Prades', '', 'Madhya Pradesh', 'Buenos Aires'
                    ]
                }
            }
        }, {
            '$project': {
                'province': '$_id', 
                '_id': 0, 
                'total': 1
            }
        }, {
            '$sort': {
                'total': -1
            }
        }, {
            '$limit': 5
        }
    ]

    result = list(collection.aggregate(pipeline))
    return jsonify(result)

@app.route('/get_total_sentiment')
def get_total_sentiment():
    pipeline = [
        {
            '$group': {
                '_id': '$sentiment', 
                'count': {
                    '$sum': 1
                }
            }
        }, {
            '$project': {
                'sentiment': '$_id', 
                '_id': 0, 
                'count': 1
            }
        }
    ]

    result = list(collection.aggregate(pipeline))
    return jsonify(result)

@app.route('/get_sentiment_count_each_date')
def get_sentiment_count_each_date():
    pipeline = [
        {
            '$addFields': {
                'year_month': {
                    '$substr': [
                        '$date_published', 0, 7
                    ]
                }
            }
        }, {
            '$group': {
                '_id': '$year_month', 
                'count': {
                    '$sum': 1
                }
            }
        }, {
            '$sort': {
                '_id': 1
            }
        }, {
            '$project': {
                '_id': 0, 
                'year_month': '$_id', 
                'count': 1
            }
        }
    ]

    result = list(collection.aggregate(pipeline))
    return jsonify(result)
    
if __name__ == '__main__':
    app.run(debug=True)
