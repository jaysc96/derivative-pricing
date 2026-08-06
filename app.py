from flask import Flask, render_template, request
import pandas as pd
from pricing import European_Option, American_Option
from api import api_bp

app = Flask(__name__)
app.register_blueprint(api_bp)

@app.route('/', methods=['GET','POST'])
def calculate_price():
    # Adding tooltips for each Greek
    tooltips = {
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
            # Resolution only. The grid's extent comes from the contract, so
            # there is no longer a minimum or maximum stock price to supply.
            dt = float(request.form['timestep'])
            SO.setFDResolution(dt=dt)

        res = SO.priceOption()

        # Every method returns the same shape, so finite differences no longer
        # need a branch of their own. Rounding happens here: the library keeps
        # full precision and the display layer decides what to show.
        res_data = {
            'Option Value': [round(res.price, 3)],
            'Delta - 𝚫': [round(res.delta, 3)],
            'Gamma - 𝚪': [round(res.gamma, 3)],
            'Theta - 𝛳': [round(res.theta, 3)],
            'Vega - 𝓋': [round(res.vega, 3)],
            'Rho - 𝛒': [round(res.rho, 3)],
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
