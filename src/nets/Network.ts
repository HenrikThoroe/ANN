import map from "../shared/map"
import range from "../shared/range"
import shuffle from "../shared/shuffle"
import sigmoid from "../shared/sigmoid"
import sigmoidDerivate from "../shared/sigmoidDerivate"
import seedrandom from "seedrandom"

export type ActivationFunctionLabel = "sigmoid"

interface NeuronResult {
    func: number
    activation: number
}

export interface TrainingSet {
    input: number[]
    target: number[]
}

export default class Network {

    // Internal

    private readonly dimension: number[]

    private biases: number[][]

    private weights: number[][][]

    private learningRate: number = 0.05

    // API

    public readonly inputNeurons: number

    public readonly hiddenLayers: number

    public readonly outputNeurons: number

    public readonly activationName: ActivationFunctionLabel

    constructor(constraints: number[], activation: ActivationFunctionLabel) {
        const rng = seedrandom("0.499214413433662")

        this.biases = map(constraints.length - 1, l => map(constraints[l + 1], () => rng()))
        this.weights = map(constraints.length - 1, l => map(constraints[l + 1], n => map(constraints[l], () => rng())))

        this.inputNeurons = constraints[0]
        this.outputNeurons = constraints[constraints.length - 1]
        this.hiddenLayers = constraints.length - 2
        this.activationName = activation
        this.dimension = [...constraints]
    }

    log() {
        console.log(this.biases)
        console.log(this.weights)
        console.log(this.hiddenLayers)
        console.log(this.inputNeurons)
        console.log(this.outputNeurons)
        console.log(this.dimension)
    }

    predict(input: number[]) {
        if (input.length !== this.dimension[0]) {
            throw new Error(`Network expects ${this.dimension[0]} input values. Given: ${input.length}`)
        }

        return this.feedforward(input)[this.dimension.length - 1]
    }

    train(epochs: number, trainingSets: TrainingSet[]) {
        for (const _e of range(epochs)) {
            for (const set of trainingSets) {
                this.backpropagate(set.input, set.target)
            }
        }
    }

    error(trainingSets: TrainingSet[]) {
        return trainingSets.map(set => this.feedforward(set.input)[this.dimension.length - 1].map((out, i) => Math.pow(out - set.target[i], 2)).reduce((p, c) => p + c)).reduce((p, c) => p + c) / trainingSets.length
    }

    // Internal

    private activation(value: number): number {
        switch (this.activationName) {
            case "sigmoid":
                return sigmoid(value)
        }
    }

    private activationDerivate(value: number): number {
        switch (this.activationName) {
            case "sigmoid":
                return sigmoidDerivate(value)
        }
    }

    private call(input: number[], layer: number, neuron: number): NeuronResult {
        // layer = layer - 1
        const func = input.map((v, i) => v * this.weights[layer][neuron][i]).reduce((p, c) => p + c) + this.biases[layer][neuron]
        const value = this.activation(func)

        return {
            func: func,
            activation: value
        }
    }

    private feedforward(data: number[]): number[][] {
        const result: number[][] = [[...data]]

        for (const layer of range(this.dimension.length - 1)) {
            result.push(map(this.dimension[layer + 1], n => this.call(result[layer], layer, n).activation))
        }

        return result
    }

    private backpropagate(input: number[], target: number[]) {
        const output = this.feedforward(input)
        const deltas: number[][][] = []
        const biasDeltas: number[][] = []

        for (let layer = output.length - 1; layer > 0; --layer) {
            let errorDerivates: number[] = []

            if (layer === output.length - 1) {
                errorDerivates = output[layer].map((o, i) => 2 * (o - target[i]))
            } else {
                errorDerivates = deltas[layer - 1].reduce((p, c) => p.map((v, i) => v + c[i]))
            }

            const input = output[layer - 1]
            const wd = errorDerivates.map((delta, n) => input.map(x => delta * this.activationDerivate(this.call(input, layer - 1, n).func) * x))
            const bd = errorDerivates.map((delta, n) => delta * this.activationDerivate(this.call(input, layer - 1, n).func))

            deltas.push(wd)
            biasDeltas.push(bd)
        }

        biasDeltas.reverse()
        deltas.reverse()

        for (const layer of range(deltas.length)) {
            for (const neuron of range(deltas[layer].length)) {
                this.biases[layer][neuron] -= this.learningRate * biasDeltas[layer][neuron]

                for (const weight of range(deltas[layer][neuron].length)) {
                    this.weights[layer][neuron][weight] -= this.learningRate * deltas[layer][neuron][weight]
                }
            }
        }
    }
}