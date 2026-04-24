from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Веса и смещения нейронной сети (те же, что в Python‑версии)
input_hidden_weights = np.array([
    [-1.88529711728749,  0.789588664900563],
    [ 0.696580163488472, -0.418275907658875],
    [-6.82879023527012,  2.32962161915203],
    [-2.64738467784819, -1.38527372034280],
    [-3.50564493311634,  0.941318558085521],
    [-2.51568815551344,  0.477977215460895],
    [-3.00093169653597,  2.05052585068828],
    [-1.41624689142756, -0.748463668009605],
    [-5.35045668177736,  0.103933183581216],
    [-0.338575234609482, 0.551327952448780],
    [-3.00658714432659,  2.14358834984641]
])

hidden_bias = np.array([
    -0.126156799473621, -0.0971099332306784, -0.0131290177129621,
    -0.816117565697241,  0.791759162494855,  -1.06118309521381,
    -0.0584932048976054, -0.131628886819008,  1.41324038475371,
     0.406437847982929, -0.123737911846614
])

hidden_output_wts = np.array([
    [0.524556095258189, 3.29400121470051, -3.71786264345922,
     -2.81427953428283, -6.98988384710971, 2.98360344445340,
     1.64788339909841, -2.97812672361198, 6.84416336975163,
     -1.83195296140828, 1.99767830404474]
])

output_bias = np.array([-0.500796073546398])

# Параметры нормализации
max_input = np.array([462.0, 1.9])
min_input = np.array([32.0, -15.5])
max_target = np.array([467.0])
min_target = np.array([85.0])
mean_inputs = np.array([246.78, -6.454])

def scale_inputs(input_data, minimum=0, maximum=1):
    """Нормализует входные данные в диапазон [minimum, maximum]"""
    delta = (maximum - minimum) / (max_input - min_input)
    scaled = minimum - delta * min_input + delta * input_data
    return scaled

def unscale_targets(output_data, minimum=0, maximum=1):
    """Денормализует выходные данные обратно в исходный диапазон"""
    delta = (maximum - minimum) / (max_target - min_target)
    unscaled = (output_data - minimum + delta * min_target) / delta
    return unscaled

def logistic(x):
    """Логистическая функция активации"""
    if x > 100.0:
        return 1.0
    elif x < -100.0:
        return 0.0
    else:
        return 1.0 / (1.0 + np.exp(-x))

def compute_feed_forward_signals(mat_inout, v_in, v_bias, size1, size2, layer):
    """Вычисляет прямой проход для слоя нейронной сети"""
    v_out = np.zeros(size2)

    for row in range(size2):
        v_out[row] = np.sum(mat_inout[row, :size1] * v_in) + v_bias[row]

        if layer == 0:  # Скрытый слой — логистическая активация
            v_out[row] = logistic(v_out[row])
        elif layer == 1:  # Выходной слой — экспоненциальная активация
            v_out[row] = np.exp(v_out[row])

    return v_out

def run_neural_net_regression(input_data):
    """Запускает нейронную сеть для регрессии"""
    # Прямой проход через скрытый слой
    hidden_layer = compute_feed_forward_signals(
        input_hidden_weights, input_data, hidden_bias, 2, 11, 0
    )

    # Прямой проход через выходной слой
    output_layer = compute_feed_forward_signals(
        hidden_output_wts, hidden_layer, output_bias, 11, 1, 1
    )

    return output_layer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получение данных из формы
        var2 = float(request.form['var2'])
        var3 = float(request.form['var3'])

        # Замена пропущенных значений на средние
        input_data = np.array([var2, var3])
        input_data[input_data == -9999] = mean_inputs[input_data == -9999]

        # Нормализация входных данных
        scaled_input = scale_inputs(input_data)

        # Запуск нейронной сети
        normalized_output = run_neural_net_regression(scaled_input)

        # Денормализация результата
        final_output = unscale_targets(normalized_output)

        return jsonify({
            'success': True,
            'prediction': f"{final_output[0]:.6f}"
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)