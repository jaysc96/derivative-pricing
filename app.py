from flask import Flask, render_template, request
import numpy as np
from src.option import European_Option, American_Option

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def calculate_price():
    if request.method == 'POST':
        S = float(request.form['stock_price'])
        K = float(request.form['strike_price'])
        T = float(request.form['time_to_expiry'])
        r = float(request.form['risk_free_rate'])
        y = float(request.form['yield_rate'])
        sig = float(request.form['volatility'])
        option_type = request.form['option_type']
        kind = request.form['kind']
        method = request.form['method']

        if kind == 'european':
            SO = European_Option(option_type, S, K, r, sig, y, T, method)
        else:
            SO = American_Option(option_type, S, K, r, sig, y, T, method)

        if method == 'MC':
            seed = int(request.form['seed'])
            n = int(request.form['iterations'])
            dt = float(request.form['timestep'])
            SO.setSeedVariables(seed=seed,n=n,dt=dt)
        elif method == 'LSMC':
            seed = int(request.form['seed'])
            n = int(request.form['iterations'])
            SO.setSeedVariables(seed=seed,n=n)
        elif method in ['BT','TT']:
            n = int(request.form['time_steps'])
            SO.setTreeSteps(n=n)
        elif method == 'FD':
            S_min = float(request.form['stock_min_price'])
            S_max = float(request.form['stock_max_price'])
            dt = float(request.form['timestep'])
            SO.setFDVariables(S_min=S_min,S_max=S_max,dt=dt)

        res = SO.priceOption()

        price = res['price']
        del res['price']
        greeks = res
        return render_template('index.html',price=price,greeks=greeks)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
