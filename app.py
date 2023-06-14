import numpy as np
from dash.exceptions import PreventUpdate
from scipy.integrate import odeint
from skfuzzy import control as ctrl
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, MATCH, State
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


'''Zdefiniowana jest funkcja tank, która reprezentuje model matematyczny zbiornika.
Przyjmuje ona argumenty: h - poziom cieczy w zbiorniku, t - czas, q_in - przepływ do zbiornika, A - przekrój zbiornika
(domyślnie ustawiony na 1.0). Funkcja oblicza przepływ q_out na podstawie równania q_out = C * sqrt(h) i zwraca wartość
zmiany poziomu cieczy w zbiorniku.'''

# Symulacja zbiornika bez regulatora
t = np.linspace(0, 100)
h0 = 0.0
q_in = 1.5
h = odeint(tank, h0, t, args=(q_in,))  # Dodanie przecinka

'''Tworzona jest symulacja zbiornika bez regulatora. Tworzony jest równomierny grid czasu t w zakresie od 0 do 100.
Ustawiany jest początkowy poziom cieczy h0 na 0.0 i przepływ do zbiornika q_in na 1.5.
Następnie wykorzystując funkcję odeint z modułu scipy.integrate, obliczane są wartości poziomu cieczy h w kolejnych
chwilach czasu.'''

# Regulator PID
K_p = 0.1
K_i = 0.1
K_d = 0.01
h_ref = 1


def pid_controller(t, h, h_ref, h_prev, e_int, dt, K_p=0.1, K_i=0.1, K_d=0.01):
    e = h_ref - h
    de = (h - h_prev) / dt
    e_int += e * dt
    q_in = K_p * e + K_i * e_int + K_d * de
    return max(0.0, float(q_in)), e, e_int  # Upewnij się, że zwracana wartość q_in jest wartością zmiennoprzecinkową


'''Zdefiniowana jest funkcja pid_controller, która implementuje algorytm regulatora PID. Przyjmuje ona argumenty:
t - czas, h - aktualny poziom cieczy, h_ref - zadany poziom cieczy, h_prev - poprzedni poziom cieczy,
e_int - całkowity błąd regulacji, dt - różnica czasu, oraz opcjonalne parametry regulatora PID: K_p, K_i, K_d.
Funkcja oblicza błąd regulacji e, pochodną błędu de, całkowity błąd regulacji e_int, oraz sygnał sterujący q_in
na podstawie algorytmu PID. Funkcja zwraca wartość sygnału sterującego q_in.'''

h0 = 0.0
h = np.zeros_like(t)
h[0] = h0
e_int = 0.0

for i in range(1, len(t)):
    dt = t[i] - t[i - 1]
    q_in, e, e_int = pid_controller(t[i], h[i - 1], h_ref, h[i - 2] if i > 1 else h[i - 1], e_int, dt)
    h[i] = odeint(tank, h[i - 1], [t[i - 1], t[i]], args=(q_in,))[1]

'''W pętli for dla każdej chwili czasu t[i] (począwszy od drugiego elementu):

Obliczana jest różnica czasu dt między bieżącym a poprzednim czasem.
Wywoływana jest funkcja pid_controller z odpowiednimi argumentami, aby obliczyć wartość sygnału sterującego q_in,
błąd regulacji e i całkowity błąd regulacji e_int. Wykorzystując funkcję odeint z modułu scipy.integrate,
obliczany jest nowy poziom cieczy h[i] w zbiorniku, na podstawie poprzedniego poziomu cieczy h[i - 1] oraz wartości
sygnału sterującego q_in. Funkcja odeint rozwiązuje równanie różniczkowe opisujące zachowanie zbiornika.
Na końcu pętli for wartość bieżącego poziomu cieczy h[i] jest zapisywana do tablicy h.'''

h0 = 0.0
h = np.zeros_like(t)
h[0] = h0
e_prev = 0.0

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True)
# SLATE
'''Tworzony jest interfejs użytkownika za pomocą biblioteki Dash. Interfejs zawiera interaktywne wykresy przedstawiające
zmiany poziomu cieczy w zbiorniku dla symulacji bez regulatora oraz z regulatorem PID. Wykresy są tworzone za pomocą
modułu plotly.graph_objs. Dodatkowo, interfejs umożliwia wprowadzanie wartości parametrów regulatora PID i zadanej
wartości poziomu cieczy.'''

app.layout = html.Div(
    [
        html.H2('Symulacja regulacji poziomu cieczy w zbiorniku',
                style={'margin': '0 auto', 'text-align': 'center', 'margin-bottom': '10px'}),

        html.Div(
            [
                html.Label('Zadany poziom cieczy (h_ref[m])',
                           style={'font-family': 'Arial', 'font-size': '20px',
                                  "margin-bottom": "5px"}),
                dbc.Input(
                    id={
                        'type': 'dynamic-input',
                        'index': 'h_ref-slider'
                    },
                    type='number',
                    placeholder='Wprowadź wartość...',
                    value=1,
                    min=0,
                    max=5,
                    size="sm",
                    style={'float': 'right', 'margin-right': "20px", "width": "200px", "background-color": "#333333",
                           "color": "white"}
                ),
                dcc.Slider(
                    id={
                        'type': 'dynamic-slider',
                        'index': 'h_ref-slider'
                    },
                    min=0,
                    max=5,
                    value=1,
                    className='slider',
                ),
            ],
            className='slider-container all-sliders'
        ),

        html.Div(
            [
                html.Label('Czas symulacji (t[s])',
                           style={'font-family': 'Arial', 'font-size': '20px',
                                  "margin-bottom": "5px"}),
                dbc.Input(
                    id={
                        'type': 'dynamic-input',
                        'index': 't-slider'
                    },
                    type='text',
                    placeholder='Wprowadź wartość...',
                    value=100,
                    min=20,
                    size="sm",
                    style={'float': 'right', 'margin-right': "20px", "width": "200px", "background-color": "#333333",
                           "color": "white"}

                ),
                dcc.Slider(
                    id={
                        'type': 'dynamic-slider',
                        'index': 't-slider'
                    },
                    min=20,
                    max=200,
                    value=100,
                    className='slider',

                ),
            ],
            className='slider-container all-sliders'
        ),

        html.Div(
            [
                html.Label('Początkowy poziom cieczy (h0[m])',
                           style={'font-family': 'Arial', 'font-size': '20px',
                                  "margin-bottom": "5px"}),
                dbc.Input(
                    id={
                        'type': 'dynamic-input',
                        'index': 'h-slider'
                    },
                    type='number',
                    placeholder='Wprowadź wartość...',
                    value=0,
                    min=0,
                    max=5,
                    size="sm",
                    style={'float': 'right', 'margin-right': "20px", "width": "200px", "background-color": "#333333",
                           "color": "white"}

                ),

                dcc.Slider(
                    id={
                        'type': 'dynamic-slider',
                        'index': 'h-slider'
                    },
                    min=0,
                    max=5,
                    value=0,
                    className='slider'
                ),
            ],
            className='slider-container all-sliders'
        ),

        html.Div(
            [
                html.Label('Przekrój zbiornika (a[m])',
                           style={'font-family': 'Arial', 'font-size': '20px',
                                  "margin-bottom": "5px"}),
                dbc.Input(
                    id={
                        'type': 'dynamic-input',
                        'index': 'a-slider'
                    },
                    type='number',
                    placeholder='Wprowadź wartość...',
                    min=0,
                    max=5,
                    value=1,
                    size="sm",
                    style={'float': 'right', 'margin-right': "20px", "width": "200px", "background-color": "#333333",
                           "color": "white"}

                ),
                dcc.Slider(
                    id={
                        'type': 'dynamic-slider',
                        'index': 'a-slider'
                    },
                    min=0,
                    max=5,
                    value=1,
                    className='slider'
                ),
            ],
            className='slider-container all-sliders'
        ),

        dbc.Button(
            "Zbiornik bez regulatora",
            id="collapse-button-bez",
            className="mb-3",
            color="dark",
            n_clicks=0,
            style={"margin": "10px", "margin-left": "35%"}
        ),

        dbc.Button(
            "Regulator PID",
            id="collapse-button",
            className="mb-3",
            color="dark",
            n_clicks=0,
            style={"margin": "10px"}
        ),

        dbc.Button(
            "Regulator Rozmyty",
            id="collapse-button-rozmyty",
            className="mb-3",
            color="dark",
            n_clicks=0,
            style={"margin": "10px"}
        ),

        dbc.Collapse(
            dbc.Card(dbc.CardBody(
                [
                    # """ WYKRES BEZ REGULATORA """
                    html.H3('Zbiornik bez regulatora',
                            style={'margin-bottom': '5px'}),

                    html.Div(
                        [
                            dcc.Graph(id='graph1',
                                      style={'display': 'inline-block', 'width': '100%', 'height': '100%'}),

                        ],
                        className='graph-container',
                        style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}
                    )
                ],
            ), color="dark"),
            id="collapse-bez",
            is_open=False,
        ),

        dbc.Collapse(
            dbc.Card(dbc.CardBody(
                [
                    html.H3('Konfiguracja regulatora PID'),

                    # """ SUWAKI PID """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label('Próbkowanie (P)',
                                                       style={'font-family': 'Arial',
                                                              'font-size': '20px',
                                                              'margin-bottom': "5px"
                                                              }),

                                            dbc.Input(
                                                id={
                                                    'type': 'dynamic-input',
                                                    'index': 'p-slider'
                                                },
                                                type='number',
                                                min=0,
                                                max=1,
                                                value=0.1,
                                                size="sm",
                                                placeholder='Wprowadź wartość...',
                                                style={'float': 'right', 'margin-right': "20px", "width": "200px",
                                                       "background-color": "#333333",
                                                       "color": "white"}
                                            ),

                                            dcc.Slider(
                                                id={
                                                    'type': 'dynamic-slider',
                                                    'index': 'p-slider'
                                                },
                                                min=0,
                                                max=1,
                                                value=0.1,
                                                className='slider'
                                            ),
                                        ],
                                        className='slider-container'
                                    ),

                                    html.Div(
                                        [
                                            html.Label('Zdwojenie regulatora (I)',
                                                       style={'font-family': 'Arial',
                                                              'font-size': '20px',
                                                              'margin-bottom': "5px"
                                                              }),

                                            dbc.Input(
                                                id={
                                                    'type': 'dynamic-input',
                                                    'index': 'i-slider'
                                                },
                                                type='number',
                                                placeholder='Wprowadź wartość...',
                                                min=0,
                                                max=1,
                                                value=0.1,
                                                size="sm",
                                                style={'float': 'right', 'margin-right': "20px", "width": "200px",
                                                       "background-color": "#333333",
                                                       "color": "white"}
                                            ),

                                            dcc.Slider(
                                                id={
                                                    'type': 'dynamic-slider',
                                                    'index': 'i-slider'
                                                },
                                                min=0,
                                                max=1,
                                                value=0.1,
                                                className='slider'
                                            ),
                                        ],
                                        className='slider-container'
                                    ),

                                    html.Div(
                                        [
                                            html.Label('Wyprzedzenie regulatora (D)',
                                                       style={'font-family': 'Arial',
                                                              'font-size': '20px',
                                                              'margin-bottom': "5px"
                                                              }),
                                            dbc.Input(
                                                id={
                                                    'type': 'dynamic-input',
                                                    'index': 'd-slider'
                                                },
                                                min=0,
                                                max=0.5,
                                                value=0.01,
                                                size="sm",
                                                type='number',
                                                placeholder='Wprowadź wartość...',
                                                style={'float': 'right', 'margin-right': "20px", "width": "200px",
                                                       "background-color": "#333333",
                                                       "color": "white"}
                                            ),

                                            dcc.Slider(
                                                id={
                                                    'type': 'dynamic-slider',
                                                    'index': 'd-slider'
                                                },
                                                min=0,
                                                max=0.5,
                                                value=0.01,
                                                className='slider'
                                            ),

                                        ],
                                        className='slider-container',
                                    ),
                                ],
                                style={'display': 'inline-block', 'height': '100%', 'width': '30%'}
                            ),

                            dcc.Graph(id='graph2', style={'display': 'inline-block', 'height': '100%', 'width': '70%'}),
                        ],
                        style={'display': 'flex', 'align-items': 'center'}
                    ),
                    # """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                ]
            ), color="dark"),
            id="collapse",
            is_open=False,
        ),

        dbc.Collapse(
            dbc.Card(dbc.CardBody(
                [
                    # """ SUWAKI ROZMYTEGO """
                    html.H3('Konfiguracja regulatora rozmytego',
                            style={'margin-bottom': '20px'}),

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label('Błąd regulacji (e)',
                                                       style={'font-family': 'Arial',
                                                              'font-size': '20px',
                                                              'margin-bottom': '5px'}),
                                            dbc.Input(
                                                id={
                                                    'type': 'dynamic-input',
                                                    'index': 'e-slider'
                                                },
                                                min=4,
                                                max=20,
                                                value=8,
                                                type='number',
                                                size="sm",
                                                placeholder='Wprowadź wartość...',
                                                style={'float': 'right', 'margin-right': "20px", "width": "200px",
                                                       "background-color": "#333333",
                                                       "color": "white"}
                                            ),

                                            dcc.Slider(
                                                id={
                                                    'type': 'dynamic-slider',
                                                    'index': 'e-slider'
                                                },
                                                min=4,
                                                max=20,
                                                value=8,
                                                className='slider'
                                            )

                                        ],
                                        className='slider-container fuzzy-slider'
                                    ),

                                    html.Div(
                                        [
                                            html.Label('Zmiana błędu regulacji (de)',
                                                       style={'font-family': 'Arial',
                                                              'font-size': '20px',
                                                              'margin-bottom': '5px'}),
                                            dbc.Input(
                                                id={
                                                    'type': 'dynamic-input',
                                                    'index': 'de-slider'
                                                },
                                                min=4,
                                                max=20,
                                                value=8,
                                                type='number',
                                                size="sm",
                                                placeholder='Wprowadź wartość...',
                                                style={'float': 'right', 'margin-right': "20px", "width": "200px",
                                                       "background-color": "#333333",
                                                       "color": "white"}
                                            ),

                                            dcc.Slider(
                                                id={
                                                    'type': 'dynamic-slider',
                                                    'index': 'de-slider'
                                                },
                                                min=4,
                                                max=20,
                                                value=8,
                                                className='slider'
                                            )
                                        ],
                                        className='slider-container fuzzy-slider'
                                    ),

                                    html.Div(
                                        [
                                            html.Label('Wsp. wpływu cieczy (q_in [m^3/s])',
                                                       style={'font-family': 'Arial',
                                                              'font-size': '20px',
                                                              'margin-bottom': '5px'}),
                                            dbc.Input(
                                                id={
                                                    'type': 'dynamic-input',
                                                    'index': 'q_in-slider'
                                                },
                                                min=4,
                                                max=20,
                                                value=11.7,
                                                type='number',
                                                size="sm",
                                                placeholder='Wprowadź wartość...',
                                                style={'float': 'right', 'margin-right': "20px", "width": "200px",
                                                       "background-color": "#333333",
                                                       "color": "white"}
                                            ),

                                            dcc.Slider(
                                                id={
                                                    'type': 'dynamic-slider',
                                                    'index': 'q_in-slider'
                                                },
                                                min=4,
                                                max=20,
                                                value=11.7,
                                                className='slider'
                                            )
                                        ],
                                        className='slider-container fuzzy-slider'
                                    ),
                                ],
                                style={'display': 'inline-block', 'height': '100%', 'width': '30%'}
                            ),

                            dcc.Graph(id='graph3', style={'display': 'inline-block', 'height': '100%', 'width': '70%'}),
                        ],
                        style={'display': 'flex', 'align-items': 'center'}
                    ),
                    # """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                ]
            ), color="dark"),
            id="collapse-rozmyty",
            is_open=False,
        ),
    ],
    style={'padding': "20px"}
)


@app.callback(
    Output({'type': 'dynamic-input', 'index': MATCH}, 'value'),
    Output({'type': 'dynamic-slider', 'index': MATCH}, 'value'),
    Input({'type': 'dynamic-slider', 'index': MATCH}, 'value'),
    Input({'type': 'dynamic-input', 'index': MATCH}, 'value'))
def update_slider_and_input(slider_val, input_val):
    ctx = dash.callback_context

    # Check which input fired the callback
    if ctx.triggered:
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if "dynamic-slider" in input_id:
            return str(slider_val), dash.no_update
        else:
            return input_val, float(input_val)
    else:
        return dash.no_update, dash.no_update


@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse_pid(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse-rozmyty", "is_open"),
    [Input("collapse-button-rozmyty", "n_clicks")],
    [State("collapse-rozmyty", "is_open")],
)
def toggle_collapse_rozmyty(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse-bez", "is_open"),
    [Input("collapse-button-bez", "n_clicks")],
    [State("collapse-bez", "is_open")],
)
def toggle_collapse_rozmyty(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    [
        Output('graph1', 'figure'),
        Output('graph2', 'figure'),
        Output('graph3', 'figure'),
    ],
    [
        Input({'type': 'dynamic-slider', 'index': 'h_ref-slider'}, 'value'),
        Input({'type': 'dynamic-slider', 'index': 't-slider'}, 'value'),
        Input({'type': 'dynamic-slider', 'index': 'p-slider'}, 'value'),
        Input({'type': 'dynamic-slider', 'index': 'i-slider'}, 'value'),
        Input({'type': 'dynamic-slider', 'index': 'd-slider'}, 'value'),
        Input({'type': 'dynamic-slider', 'index': 'h-slider'}, 'value'),
        Input({'type': 'dynamic-slider', 'index': 'a-slider'}, 'value'),
        Input({'type': 'dynamic-slider', 'index': 'e-slider'}, 'value'),
        Input({'type': 'dynamic-slider', 'index': 'de-slider'}, 'value'),
        Input({'type': 'dynamic-slider', 'index': 'q_in-slider'}, 'value')
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
    trace1 = go.Scatter(x=t, y=np.squeeze(h), mode='lines', name='Zbiornik bez regulatora', line=dict(color='#00B3FF'))

    # Symulacja zbiornika z regulatorem PID
    h_ref = h_ref_slider_value  # Użyj wartości z suwaka
    h = np.zeros_like(t)
    h[0] = h0
    e_int = 0.0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        q_in, e, e_int = pid_controller(t[i], h[i - 1], h_ref, h[i - 2] if i > 1 else h[i - 1], e_int, dt, p_value,
                                        i_value, d_value)
        h[i] = odeint(tank, h[i - 1], [t[i - 1], t[i]], args=(q_in, a_value,))[1]  # Dodanie przecinka
    trace2 = go.Scatter(x=t, y=h, mode='lines', name='Zbiornik z regulatorem PID', line=dict(color='#FF0000'))

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
    trace3 = go.Scatter(x=t, y=h, mode='lines', name='Zbiornik z regulatorem rozmytym', line=dict(color='#00ff03'))

    layout1 = go.Layout(xaxis=dict(title='Czas [s]'),
                        yaxis=dict(title='Poziom wody [m]'), plot_bgcolor='#272b30', paper_bgcolor='#272b30',
                        font=dict(color='#ffffff'))

    layout2 = go.Layout(xaxis=dict(title='Czas [s]'),
                        yaxis=dict(title='Poziom wody [m]'), plot_bgcolor='#272b30', paper_bgcolor='#272b30',
                        font=dict(color='#ffffff'))

    layout3 = go.Layout(xaxis=dict(title='Czas [s]'),
                        yaxis=dict(title='Poziom wody [m]'), plot_bgcolor='#272b30', paper_bgcolor='#272b30',
                        font=dict(color='#ffffff'))

    return go.Figure(data=[trace1], layout=layout1), go.Figure(data=[trace2], layout=layout2), go.Figure(
        data=[trace3], layout=layout3)


if __name__ == '__main__':
    app.run_server(debug=True)
