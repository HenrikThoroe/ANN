import NeuralNetwork from "./NeuralNetwork";
import Neuron from "./Neuron";

const neuron = new Neuron(2)

const net = new NeuralNetwork()

const data: [number[], number][] = [
    [[2, 0], 0],
    [[0, 2], 1],
    [[2, 2], 1],
    [[0, 0], 0],
    [[1, 0], 1]
]

// for (let i = 0; i < 1000000; ++i) {
//     data.forEach(d => {
//         net.train(d[0], [d[1]])
//     })
// } 

// data.forEach(d => {
//     console.log(net.feedforward(d[0]))
// })

for (let i = 0; i < 1000000; ++i) {
    data.forEach(d => {
        neuron.train(d[0], d[1])
    })
}

data.forEach(d => {
    console.log(neuron.predict(d[0]))
})