from flask import Flask, render_template, request
import numpy as np
from src.option import Stock_Option

app = Flask(__name__)

@app.route('/pricing', methods=['GET','POST'])
def calculate_price():
    if request.method == 'POST':
        S = float(request.form['stock_price'])
        K = float(request.form['strike_price'])
        T = float(request.form['time_to_expiry'])
        r = float(request.form['risk_free_rate'])
        y = float(request.form['yield_rate'])
        sig = float(request.form['volatility'])
        option_type = request.form['option_type']

        SO = Stock_Option(option_type,S,K,r,sig,y,T)#,kind,start_date,end_date)
        price = SO.priceOption()

        return render_template('index.html',price=price)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
