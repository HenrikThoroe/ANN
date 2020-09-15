import Neuron from "./Neuron";

const neuron = new Neuron(2)

const data: [number[], number][] = [
    [[1, 0], 1],
    [[0, 1], 0],
    [[1, 1], 1],
    [[0, 0], 0]
]

for (let i = 0; i < 1000000; ++i) {
    data.forEach(d => {
        neuron.train(d[0], d[1])
    })
}

data.forEach(d => {
    console.log(neuron.predict(d[0]))
})