from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from src.option import European_Option, American_Option

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def calculate_price():
    # Adding tooltips for each Greek
    tooltips = {
        'Stock Price': 'Initial stock price',
        'Option Value': 'Option value based on the chosen model',
        'Delta 𝚫': 'Rate of change of option value with respect to the underlying asset price',
        'Gamma 𝚪': 'Rate of change of delta with respect to the underlying asset price',
        'Theta 𝚯': 'Rate of change of option value with respect to time',
        'Vega 𝓋': 'Option sensitivity to volatility of the underlying asset',
        'Rho 𝛒': 'Option sensitivity to interest rate changes'
    }

    if request.method == 'POST':
        S = float(request.form['stock_price'])
        K = float(request.form['strike_price'])
        T = float(request.form['time_to_expiry'])
        r = float(request.form['risk_free_rate'])
        y = float(request.form['yield_rate'])
        sig = float(request.form['volatility'])
        option_type = request.form['option_type']
        exercise_type = request.form['exercise_type']
        method = request.form['method']

        if exercise_type == 'european':
            SO = European_Option(option_type, S, K, r, sig, y, T, method)
        else:
            SO = American_Option(option_type, S, K, r, sig, y, T, method)

        if method == 'MC':
            seed = int(request.form['seed'])
            n = int(request.form['iterations'])
            dt = float(request.form['timestep'])
            SO.setSeedVariables(seed=seed, n=n, dt=dt)
        elif method == 'LSMC':
            seed = int(request.form['seed'])
            n = int(request.form['iterations'])
            SO.setSeedVariables(seed=seed, n=n)
        elif method in ['BT','TT']:
            n = int(request.form['time_steps'])
            SO.setTreeSteps(n=n)
        elif method == 'FD':
            S_min = float(request.form['stock_min_price'])
            S_max = float(request.form['stock_max_price'])
            dt = float(request.form['timestep'])
            SO.setFDVariables(S_min=S_min, S_max=S_max, dt=dt)

            res = SO.priceOption()
            res_data = {
                'Stock Price': res['stock_price'], 
                'Option Value': res['price'], 
                'Delta - 𝚫': res['delta'], 
                'Gamma - 𝚪': res['gamma'], 
                'Theta - 𝛳': res['theta'], 
                'Vega - 𝓋': res['vega'], 
                'Rho - 𝛒': res['rho']
            }
            df = pd.DataFrame(res_data).sort_values(by='Stock Price')
            res_table_html = df.to_html(
                classes='table table-hover table-striped table-bordered',
                index=False,
                escape=False,
                header=True
            )

            # Render the HTML table in the template
            return render_template('index.html', res_table_html=res_table_html, tooltips=tooltips)

        res = SO.priceOption()
        # return render_template('index.html',res=res)

        # Create a pandas DataFrame for the results, using the metrics as columns
        res_data = {
            'Option Value': [res['price']],
            'Delta - 𝚫': [res['delta']],
            'Gamma - 𝚪': [res['gamma']],
            'Theta - 𝛳': [res['theta']],
            'Vega - 𝓋': [res['vega']],
            'Rho - 𝛒': [res['rho']],
        }
        df = pd.DataFrame(res_data)

        # Convert DataFrame to HTML without row index
        res_table_html = df.to_html(
            classes='table table-hover table-striped table-bordered',
            index=False,
            escape=False,
        )

        # Render the HTML table in the template
        return render_template('index.html', res_table_html=res_table_html, tooltips=tooltips)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
