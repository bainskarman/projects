import dash
from dash import dcc, html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Iframe(
        src="https://app.powerbi.com/view?r=eyJrIjoiNmRkZTJkNGItMjQ5Ny00MzMzLThjMjQtYzU1YzQzZWZlNjYxIiwidCI6ImI2NDE3Y2QwLTFmNzMtNDQ3MS05YTM5LTIwOTUzODIyYTM0YSIsImMiOjN9",
        width="100%",
        height="600",
        style={"border": "none"}  # This removes the border around the iframe
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
