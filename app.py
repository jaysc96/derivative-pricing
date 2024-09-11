from flask import Flask, render_template, request
import numpy as np
from src.option import European_Option, American_Option
import sys
print(sys.executable)

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
        show_greeks = request.form.get('showGreeks') == 'on'

        SO = European_Option(option_type,S,K,r,sig,y,T)

        if method == 'MC':
            seed = int(request.form['seed'])
            n = int(request.form['iterations'])
            dt = float(request.form['timestep'])
            SO.setSeedVariables(seed=seed,n=n,dt=dt)

        res = SO.priceOption(method=method, greeks=show_greeks)
        if show_greeks:
            price = res['price']
            del res['price']
            greeks = res
            return render_template('index.html',price=price,greeks=greeks)
        return render_template('index.html',price=res)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
