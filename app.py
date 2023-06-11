import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

# Parametry zbiornika
A = 1.0  # przekrój zbiornika
C = 1.0  # stała


# Model zbiornika
def tank(h, t, q_in):
    if h < 0:
        h = 0.0
    q_out = C * np.sqrt(h)
    return (float(q_in) - q_out) / A  # Upewnij się, że q_in jest wartością zmiennoprzecinkową



# Symulacja zbiornika bez regulatora
t = np.linspace(0, 100)
h0 = 0.0
q_in = 1.5
h = odeint(tank, h0, t, args=(q_in,))  # Dodanie przecinka

# Regulator PID
K_p = 0.1
K_i = 0.1
K_d = 0.01
h_ref = 1


def pid_controller(t, h, h_ref, h_prev, e_int, dt):
    e = h_ref - h
    e_der = (h - h_prev) / dt
    e_int += e * dt
    q_in = K_p * e + K_i * e_int + K_d * e_der
    return max(0.0, float(q_in)), e, e_int  # Upewnij się, że zwracana wartość q_in jest wartością zmiennoprzecinkową


h0 = 0.0
h = np.zeros_like(t)
h[0] = h0
e_int = 0.0

for i in range(1, len(t)):
    dt = t[i] - t[i - 1]
    q_in, e, e_int = pid_controller(t[i], h[i - 1], h_ref, h[i - 2] if i > 1 else h[i - 1], e_int, dt)
    h[i] = odeint(tank, h[i - 1], [t[i - 1], t[i]], args=(q_in,))[1]

# Regulator rozmyty
# Definiowanie zbiorów rozmytych
e = ctrl.Antecedent(np.arange(-1, 12, 0.01), 'e')  # zwiększony zakres
de = ctrl.Antecedent(np.arange(-1, 10, 0.01), 'de')  # zwiększony zakres
q_in = ctrl.Consequent(np.arange(-1, 12, 0.01), 'q_in')  # zwiększony zakres

e.automf(7)  # Więcej funkcji przynależności
de.automf(7)  # Więcej funkcji przynależności
q_in.automf(7)  # Więcej funkcji przynależności

# Zdefiniuj więcej reguł dla większej liczby funkcji przynależności
rule1 = ctrl.Rule(e['poor'] & de['poor'], q_in['poor'])
rule2 = ctrl.Rule(e['mediocre'] & de['mediocre'], q_in['mediocre'])
rule3 = ctrl.Rule(e['average'] & de['average'], q_in['average'])
rule4 = ctrl.Rule(e['decent'] & de['decent'], q_in['decent'])
rule5 = ctrl.Rule(e['good'] & de['good'], q_in['good'])
rule6 = ctrl.Rule(e['excellent'] & de['excellent'], q_in['excellent'])
rule7 = ctrl.Rule(e['dismal'] & de['dismal'], q_in['dismal'])

control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
controller = ctrl.ControlSystemSimulation(control_system)

h0 = 0.0
h = np.zeros_like(t)
h[0] = h0
e_prev = 0.0

for i in range(1, len(t)):
    e = h_ref - h[i - 1]
    de = (e - e_prev) / dt
    controller.input['e'] = e
    controller.input['de'] = de
    controller.compute()
    q_in = controller.output['q_in']
    h[i] = odeint(tank, h[i - 1], [t[i - 1], t[i]], args=(q_in,))[1]
    e_prev = e


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Symulacja systemu sterowania zbiornikiem'),

    html.Div([
        html.Label('Wpływ wody'),
        dcc.Slider(
            id='q_in-slider',
            min=0,
            max=10,
            step=0.1,
            value=1.5
        )
    ]),

    html.Div([
        html.Label('Zadany poziom wody'),
        dcc.Slider(
            id='h_ref-slider',
            min=0,
            max=5,
            step=0.1,
            value=1,
        )
    ]),

    html.Div([
        html.Label('Czas symulacji'),
        dcc.Slider(
            id='t-slider',
            min=20,
            max=200,
            step=10,
            value=100,
        )
    ]),

    html.Div(children=[
        dcc.Graph(id='graph1', style={'display': 'inline-block', 'width': '60vh'}),
        dcc.Graph(id='graph2', style={'display': 'inline-block', 'width': '60vh'}),
        dcc.Graph(id='graph3', style={'display': 'inline-block', 'width': '60vh'})
    ])
])


@app.callback(
    [Output('graph1', 'figure'),
     Output('graph2', 'figure'),
     Output('graph3', 'figure')],
    [Input('q_in-slider', 'value'),
     Input('h_ref-slider', 'value'),
     Input('t-slider', 'value')]
)
def update_graph(q_in_slider_value, h_ref_slider_value, t_value):
    # Symulacja zbiornika bez regulatora
    t = np.linspace(0, t_value)
    q_in = q_in_slider_value  # Użyj wartości z suwaka
    h = odeint(tank, h0, t, args=(q_in,))
    trace1 = go.Scatter(x=t, y=np.squeeze(h), mode='lines', name='Zbiornik bez regulatora')

    # Symulacja zbiornika z regulatorem PID
    h_ref = h_ref_slider_value  # Użyj wartości z suwaka
    h = np.zeros_like(t)
    h[0] = h0
    e_int = 0.0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        q_in, e, e_int = pid_controller(t[i], h[i - 1], h_ref, h[i - 2] if i > 1 else h[i - 1], e_int, dt)
        h[i] = odeint(tank, h[i - 1], [t[i - 1], t[i]], args=(q_in,))[1]  # Dodanie przecinka
    trace2 = go.Scatter(x=t, y=h, mode='lines', name='Zbiornik z regulatorem PID')

    # Symulacja zbiornika z regulatorem rozmytym
    h = np.zeros_like(t)
    h[0] = h0
    e_prev = 0.0
    for i in range(1, len(t)):
        e = h_ref - h[i - 1]
        de = (e - e_prev) / dt
        controller.input['e'] = e
        controller.input['de'] = de
        controller.compute()
        q_in = float(controller.output['q_in'])
        h[i] = odeint(tank, h[i - 1], [t[i - 1], t[i]], args=(q_in,))[1]  # Dodanie przecinka
        e_prev = e
    trace3 = go.Scatter(x=t, y=h, mode='lines', name='Zbiornik z regulatorem rozmytym')

    return (
        {'data': [trace1], 'layout': go.Layout(title='Zbiornik bez regulatora')},
        {'data': [trace2], 'layout': go.Layout(title='Zbiornik z regulatorem PID')},
        {'data': [trace3], 'layout': go.Layout(title='Zbiornik z regulatorem rozmytym')}
    )


if __name__ == '__main__':
    app.run_server(debug=True)
