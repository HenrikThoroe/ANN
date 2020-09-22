import NeuralNetwork from "./NeuralNetwork";
import Neuron from "./Neuron";
import sigmoid from "./shared/sigmoid";
import sigmoidDerivate from "./shared/sigmoidDerivate";

const neuron = new Neuron(2)

const net = new NeuralNetwork()

const data: [number[], number][] = [
    [[1, 0], 1],
    [[0, 1], 1],
    [[0, 0], 0],
    [[1, 1], 1]
]

const neurons = [2, 1]
const biases = [[-0.16595599, 0.44064899], [-0.62747958]]
const weights = [[[-0.99977125, -0.70648822], [-0.39533485, -0.81532281]], [[-0.30887855, -0.20646505]]]

function trigger(layer: number, neuron: number, data: number[]) {
    const w: number[] = weights[layer][neuron]
    const b: number = biases[layer][neuron]
    const z = data.map((d, i) => d * w[i]).reduce((p, c) => p + c) + b
    const a = sigmoid(z)

    return [z, a]
}

function feedforward(data: number[]): number[][] {
    const result: number[][] = [data]

    for (let l = 0; l < 2; ++l) {
        const out: number[] = []

        for (let i = 0; i < neurons[l]; ++i) {
            out.push(trigger(l, i, data)[1])
        }

        result.push(out)
        data = [...out]
    }

    return result
}

const map = <T>(num: number, action: (num: number) => T): T[] => {
    const out: T[] = []
    for (let i = 0; i < num; ++i) {
        out.push(action(i))
    }
    return out
}

function backpropagate(data: number[], target: number[]) {
    const [input_layer_output, hidden_layer_output, output_layer_output] = feedforward(data)

    const error = output_layer_output.map((o, i) => Math.pow(o - target[i], 2))
    const delta = map(2, n => 2 * (output_layer_output[0] - target[0]) * sigmoidDerivate(trigger(1, 0, hidden_layer_output)[0]) * trigger(0, n, input_layer_output)[1])
    const hiddenError = map(2, n => map(2, w => (error[0] * delta[n]) * sigmoidDerivate(trigger(0, n, input_layer_output)[0]) * input_layer_output[w]))

    // console.log(error, delta, hiddenError)
    weights[0][0][0] += -0.1 * hiddenError[0][0]
    weights[0][0][1] += -0.1 * hiddenError[0][1] 
    weights[0][1][0] += -0.1 * hiddenError[1][0]
    weights[0][1][1] += -0.1 * hiddenError[1][1]
    weights[1][0][0] += -0.1 * delta[0]
    weights[1][0][1] += -0.1 * delta[1]
}

data.forEach(d => {
    console.log(feedforward(d[0])[2])
})

for (let i = 0; i < 1000000; ++i) {
    data.forEach(d => {
        backpropagate(d[0], [d[1]])
    })
}

data.forEach(d => {
    console.log(feedforward(d[0])[2])
})

// console.log("Network")

// for (let i = 0; i < 1; ++i) {
//     data.forEach(d => {
//         net.train(d[0], [d[1]])
//     })
// } 

// data.forEach(d => {
//     console.log(net.feedforward(d[0]))
// })

// console.log("Neuron")

// for (let i = 0; i < 1000000; ++i) {
//     data.forEach(d => {
//         neuron.train(d[0], d[1])
//     })
// }

// data.forEach(d => {
//     console.log(neuron.predict(d[0]))
// })