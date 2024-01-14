# Import the libraries
from dash import Dash, dcc, html, Input, Output, State
import requests
from urllib.parse import quote

# Create the app
app = Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("GramAI", style={"text-align": "center"}),
    dcc.Textarea(id="query-box", value="", placeholder="He have a car", style={"font-size": "20px", "width": "55%", "margin-top": "60px", "resize":"none"}, maxLength=200),
    html.Div(id="output", style={"font-size": "20px", "width": "55%", "margin-top": "20px", 'whiteSpace': 'pre-line'}),
    html.Button("Correct", id="update-button", style={"font-size": "20px", "width": "10%", "margin-top": "20px"}), # Add the button element
    html.Div(id="logo", style={"background-image": f"url({app.get_asset_url('dtu.png')})"}),
    html.Div(id="background", style={"background-image": f"url({app.get_asset_url('parrot.jpeg')})"}),
])

# Define the callback function
@app.callback(
    Output("output", "children"),
    Input("update-button", "n_clicks"),
    # Input("query-bar", "value")
    State("query-box", "value") # Use the input element as a state
)
def display_output(n_clicks, value):
    if value is None:
        return "No input"
    else:
        response = requests.post('https://gramai-app-h2yv3342wq-ew.a.run.app/text/?input_sentence='+quote(value))
        return response.json()['corrected']

# Run the app
if __name__ == "__main__":
    app.run(debug=True)