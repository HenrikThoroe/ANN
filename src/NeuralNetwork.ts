import Neuron from "./Neuron"
import dot from "./shared/dot"
import sigmoidDerivate from "./shared/sigmoidDerivate"

export type Layer = Neuron[]

export default class NeuralNetwork {

    private layers: Layer[]

    private readonly learningRate: number = 0.05

    constructor() {
        this.layers = [[new Neuron(2), new Neuron(2), new Neuron(2), new Neuron(2), new Neuron(2)], [new Neuron(2)]]
    }

    feedforward(input: number[]): number[] {
        return this.feed(input, 0)
    }

    train(input: number[], target: number[]) {
        const outputs = this.feedforward(input)
        const hiddenOutput = this.layers[0].map(neuron => neuron.predict(input))
        const outErrors = outputs.map((output, index) => sigmoidDerivate(output) * (output - target[index]))
        const hiddenErrors = hiddenOutput.map((hiddenOutput, index) => sigmoidDerivate(hiddenOutput) * dot([outErrors[0], outErrors[0]], this.layers[1][0].weights))

        this.layers[1].forEach((neuron, index) => {
            for (let w = 0; w <= neuron.weights.length; ++w) {
                if (w === neuron.weights.length) {
                    neuron.bias += -this.learningRate * outErrors[index]
                } else {
                    neuron.weights[w] += -this.learningRate * hiddenOutput[w] * outErrors[index]
                }
            }
        })

        this.layers[0].forEach((neuron, index) => {
            for (let w = 0; w <= neuron.weights.length; ++w) {
                if (w === neuron.weights.length) {
                    neuron.bias += -this.learningRate * hiddenErrors[index]
                } else {
                    neuron.weights[w] += -this.learningRate * input[w] * hiddenErrors[index]
                }
            }
        })
    }

    private feed(input: number[], layerIndex: number): number[] {
        const output: number[] = []

        for (const neuron of this.layers[layerIndex]) {
            output.push(neuron.predict(input))
        }

        if (layerIndex === this.layers.length - 1) {
            return output
        } else {
            return this.feed(output, layerIndex + 1)
        }
    }

}