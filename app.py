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
C = 1.0  # stała


# Model zbiornika
def tank(h, t, q_in, A=1.0):
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


def pid_controller(t, h, h_ref, h_prev, e_int, dt, K_p=0.1, K_i=0.1, K_d=0.01):
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



h0 = 0.0
h = np.zeros_like(t)
h[0] = h0
e_prev = 0.0

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Symulacja systemu sterowania zbiornikiem'),

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

    html.Div([
        html.Label('P'),
        dcc.Slider(
            id='p-slider',
            min=0,
            max=1,
            step=0.05,
            value=0.1,
        )
    ]),

    html.Div([
        html.Label('I'),
        dcc.Slider(
            id='i-slider',
            min=0,
            max=1,
            step=0.05,
            value=0.1,
        )
    ]),

    html.Div([
        html.Label('D'),
        dcc.Slider(
            id='d-slider',
            min=0,
            max=0.5,
            step=0.01,
            value=0.01,
        )
    ]),

    html.Div([
        html.Label('Początkowy poziom wody'),
        dcc.Slider(
            id='h-slider',
            min=0,
            max=5,
            step=0.1,
            value=0
        )
    ]),

    html.Div([
        html.Label('Przekrój zbiornika'),
        dcc.Slider(
            id='a-slider',
            min=0,
            max=5,
            step=0.1,
            value=1
        )
    ]),

    html.Div([
        html.Label('e'),
        dcc.Slider(
            id='e-slider',
            min=5,
            max=20,
            step=0.5,
            value=10
        )
    ]),

    html.Div([
        html.Label('de'),
        dcc.Slider(
            id='de-slider',
            min=5,
            max=20,
            step=0.5,
            value=10
        )
    ]),

    html.Div([
        html.Label('Współczynnik wpływu wody'),
        dcc.Slider(
            id='q_in-slider',
            min=5,
            max=20,
            step=0.5,
            value=10
        )
    ]),

    html.Div(children=[
        dcc.Graph(id='graph1', style={'display': 'inline-block', 'width': '55vh'}),
        dcc.Graph(id='graph2', style={'display': 'inline-block', 'width': '55vh'}),
        dcc.Graph(id='graph3', style={'display': 'inline-block', 'width': '55vh'})
    ])
])


@app.callback(
    [Output('graph1', 'figure'),
     Output('graph2', 'figure'),
     Output('graph3', 'figure')],
    [
     Input('h_ref-slider', 'value'),
     Input('t-slider', 'value'),
     Input('p-slider', 'value'),
     Input('i-slider', 'value'),
     Input('d-slider', 'value'),
     Input('h-slider', 'value'),
     Input('a-slider', 'value'),
     Input('e-slider', 'value'),
     Input('de-slider', 'value'),
     Input('q_in-slider', 'value'),
     ]
)
def update_graph(h_ref_slider_value, t_value, p_value,
                 i_value, d_value, h_value, a_value, e_slider, de_slider, q_in_slider):
    # Regulator rozmyty
    # Definiowanie zbiorów rozmytych
    e = ctrl.Antecedent(np.arange(-1, e_slider, 0.01), 'e')  # zwiększony zakres
    de = ctrl.Antecedent(np.arange(-1, de_slider, 0.01), 'de')  # zwiększony zakres
    q_in_ctr = ctrl.Consequent(np.arange(-1, q_in_slider, 0.01), 'q_in')  # zwiększony zakres

    e.automf(7)  # Więcej funkcji przynależności
    de.automf(7)  # Więcej funkcji przynależności
    q_in_ctr.automf(7)  # Więcej funkcji przynależności

    # Zdefiniuj więcej reguł dla większej liczby funkcji przynależności
    rule1 = ctrl.Rule(e['poor'] & de['poor'], q_in_ctr['poor'])
    rule2 = ctrl.Rule(e['mediocre'] & de['mediocre'], q_in_ctr['mediocre'])
    rule3 = ctrl.Rule(e['average'] & de['average'], q_in_ctr['average'])
    rule4 = ctrl.Rule(e['decent'] & de['decent'], q_in_ctr['decent'])
    rule5 = ctrl.Rule(e['good'] & de['good'], q_in_ctr['good'])
    rule6 = ctrl.Rule(e['excellent'] & de['excellent'], q_in_ctr['excellent'])
    rule7 = ctrl.Rule(e['dismal'] & de['dismal'], q_in_ctr['dismal'])

    control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
    controller = ctrl.ControlSystemSimulation(control_system)

    # Symulacja zbiornika bez regulatora
    t = np.linspace(0, t_value)
    q_in = q_in_slider  # Użyj wartości z suwaka
    h0 = h_value
    h = odeint(tank, h0, t, args=(q_in, a_value,))
    trace1 = go.Scatter(x=t, y=np.squeeze(h), mode='lines', name='Zbiornik bez regulatora')

    # Symulacja zbiornika z regulatorem PID
    h_ref = h_ref_slider_value  # Użyj wartości z suwaka
    h = np.zeros_like(t)
    h[0] = h0
    e_int = 0.0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        q_in, e, e_int = pid_controller(t[i], h[i - 1], h_ref, h[i - 2] if i > 1 else h[i - 1], e_int, dt,  p_value, i_value, d_value)
        h[i] = odeint(tank, h[i - 1], [t[i - 1], t[i]], args=(q_in, a_value,))[1]  # Dodanie przecinka
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
        h[i] = odeint(tank, h[i - 1], [t[i - 1], t[i]], args=(q_in, a_value,))[1]  # Dodanie przecinka
        e_prev = e
    trace3 = go.Scatter(x=t, y=h, mode='lines', name='Zbiornik z regulatorem rozmytym')

    return (
        {'data': [trace1], 'layout': go.Layout(title='Zbiornik bez regulatora')},
        {'data': [trace2], 'layout': go.Layout(title='Zbiornik z regulatorem PID')},
        {'data': [trace3], 'layout': go.Layout(title='Zbiornik z regulatorem rozmytym')}
    )


if __name__ == '__main__':
    app.run_server(debug=True)
