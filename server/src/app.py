from flask import Flask, jsonify, request
import numpy as np
from option import Stock_Option

app = Flask(__name__)

@app.route('/pricing', methods=['POST'])
def calculate_price():
    data = request.json

    type = data['type']
    S0 = data['initial_price']
    X = data['strike_price']
    r = data['interest_rate']
    sig = data['volatility']
    y = data['dividend_rate']
    T = data['time']
    kind = data['kind']
    start_date = data['start_date']
    end_date = data['end_date']

    SO = Stock_Option(type,S0,X,r,sig,y,T,kind,start_date,end_date)
    price = SO.priceOption()

    return jsonify({"price": price})

if __name__ == '__main__':
    app.run(debug=True)
