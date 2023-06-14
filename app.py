import numpy as np
from scipy.integrate import odeint
from skfuzzy import control as ctrl
import dash
from dash import dcc
from dash import html
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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

'''Tworzony jest interfejs użytkownika za pomocą biblioteki Dash. Interfejs zawiera interaktywne wykresy przedstawiające
zmiany poziomu cieczy w zbiorniku dla symulacji bez regulatora oraz z regulatorem PID. Wykresy są tworzone za pomocą
modułu plotly.graph_objs. Dodatkowo, interfejs umożliwia wprowadzanie wartości parametrów regulatora PID i zadanej
wartości poziomu cieczy.'''

app.layout = html.Div(
    [
        html.H1('Symulacja regulatora poziomu cieczy w zbiorniku',
                style={'color': '#00b3ff', 'margin': '0 auto', 'text-align': 'center', 'margin-bottom': '20px'}),

        html.Div(
            [
                html.Label('Zadany poziom cieczy (h_ref[m])',
                           style={'color': '#c1c1d7', 'font-family': 'Arial', 'font-size': '20px',
                                  'font-style': 'italic'}),
                dcc.Slider(
                    id='h_ref-slider',
                    min=0,
                    max=5,
                    step=0.1,
                    value=1,
                    className='slider',
                )
            ],
            className='slider-container'
        ),

        html.Div(
            [
                html.Label('Czas symulacji (t[s])', style={'color': '#c1c1d7', 'font-family': 'Arial', 'font-size': '20px',
                                                        'font-style': 'italic'}),
                dcc.Slider(
                    id='t-slider',
                    min=20,
                    max=200,
                    step=10,
                    value=100,
                    className='slider'
                )
            ],
            className='slider-container'
        ),

        html.Div(
            [
                html.Label('Współczynnik proporcjonalny (P)',
                           style={'color': '#c1c1d7', 'font-family': 'Arial', 'font-size': '20px',
                                  'font-style': 'italic'}),
                dcc.Slider(
                    id='p-slider',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.1,
                    className='slider'
                )
            ],
            className='slider-container'
        ),

        html.Div(
            [
                html.Label('Współczynnik całkujący (I)',
                           style={'color': '#c1c1d7', 'font-family': 'Arial', 'font-size': '20px',
                                  'font-style': 'italic'}),
                dcc.Slider(
                    id='i-slider',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.1,
                    className='slider'
                )
            ],
            className='slider-container'
        ),

        html.Div(
            [
                html.Label('Współczynnik różniczkujący (D)',
                           style={'color': '#c1c1d7', 'font-family': 'Arial', 'font-size': '20px',
                                  'font-style': 'italic'}),
                dcc.Slider(
                    id='d-slider',
                    min=0,
                    max=0.5,
                    step=0.01,
                    value=0.01,
                    className='slider'
                )
            ],
            className='slider-container'
        ),

        html.Div(
            [
                html.Label('Początkowy poziom cieczy (h0[m])',
                           style={'color': '#c1c1d7', 'font-family': 'Arial', 'font-size': '20px',
                                  'font-style': 'italic'}),
                dcc.Slider(
                    id='h-slider',
                    min=0,
                    max=5,
                    step=0.1,
                    value=0,
                    className='slider'
                )
            ],
            className='slider-container'
        ),

        html.Div(
            [
                html.Label('Przekrój zbiornika (a[m])',
                           style={'color': '#c1c1d7', 'font-family': 'Arial', 'font-size': '20px',
                                  'font-style': 'italic'}),
                dcc.Slider(
                    id='a-slider',
                    min=0,
                    max=5,
                    step=0.1,
                    value=1,
                    className='slider'
                )
            ],
            className='slider-container'
        ),

        html.Div(
            [
                html.Label('Błąd regulacji (e)', style={'color': '#c1c1d7', 'font-family': 'Arial', 'font-size': '20px',
                                                        'font-style': 'italic'}),
                dcc.Slider(
                    id='e-slider',
                    min=5,
                    max=20,
                    step=0.5,
                    value=10,
                    className='slider'
                )
            ],
            className='slider-container'
        ),

        html.Div(
            [
                html.Label('Zmiana błędu regulacji (de)',
                           style={'color': '#c1c1d7', 'font-family': 'Arial', 'font-size': '20px',
                                  'font-style': 'italic'}),
                dcc.Slider(
                    id='de-slider',
                    min=5,
                    max=20,
                    step=0.5,
                    value=10,
                    className='slider'
                )
            ],
            className='slider-container'
        ),

        html.Div(
            [
                html.Label('Współczynnik wpływu cieczy (q_in [m^3/s])',
                           style={'color': '#c1c1d7', 'font-family': 'Arial', 'font-size': '20px',
                                  'font-style': 'italic'}),
                dcc.Slider(
                    id='q_in-slider',
                    min=5,
                    max=20,
                    step=0.5,
                    value=10,
                    className='slider'
                )
            ],
            className='slider-container'
        ),

        html.Div(
            [
                dcc.Graph(id='graph1', style={'display': 'inline-block', 'width': '33%', 'height': '100%'}),
                dcc.Graph(id='graph2', style={'display': 'inline-block', 'width': '33%', 'height': '100%'}),
                dcc.Graph(id='graph3', style={'display': 'inline-block', 'width': '33%', 'height': '100%'})
            ],
            className='graph-container',
            style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}
        )
    ],
    style={'backgroundColor': '#212529', 'padding': "10px"}
)


@app.callback(
    [
        Output('graph1', 'figure'),
        Output('graph2', 'figure'),
        Output('graph3', 'figure')
    ],
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
        Input('q_in-slider', 'value')
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
    trace1 = go.Scatter(x=t, y=np.squeeze(h), mode='lines', name='Zbiornik bez regulatora', line=dict(color='#FF0000'))

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
    trace3 = go.Scatter(x=t, y=h, mode='lines', name='Zbiornik z regulatorem rozmytym', line=dict(color='#FF0000'))

    layout1 = go.Layout(title='Zbiornik bez regulatora', xaxis=dict(title='Czas [s]'),
                        yaxis=dict(title='Poziom wody [m]'), plot_bgcolor='#212529', paper_bgcolor='#212529',
                        font=dict(color='#ffffff'))

    layout2 = go.Layout(title='Zbiornik z regulatorem PID', xaxis=dict(title='Czas [s]'),
                        yaxis=dict(title='Poziom wody [m]'), plot_bgcolor='#212529', paper_bgcolor='#212529',
                        font=dict(color='#ffffff'))

    layout3 = go.Layout(title='Zbiornik z regulatorem rozmytym', xaxis=dict(title='Czas [s]'),
                        yaxis=dict(title='Poziom wody [m]'), plot_bgcolor='#212529', paper_bgcolor='#212529',
                        font=dict(color='#ffffff'))

    return go.Figure(data=[trace1], layout=layout1), go.Figure(data=[trace2], layout=layout2), go.Figure(
        data=[trace3], layout=layout3)


if __name__ == '__main__':
    app.run_server(debug=False)
