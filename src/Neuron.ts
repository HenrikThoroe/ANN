import dot from "./shared/dot"
import sigmoid from "./shared/sigmoid"
import sigmoidDerivate from "./shared/sigmoidDerivate"

export default class Neuron {

    private readonly learningRate: number = 0.05

    private readonly size: Number

    private bias: number

    private weights: number[]

    constructor(size: number) {
        this.size = size
        this.bias = 1
        this.weights = new Array(size)

        for (let i = 0; i < size; ++i) {
            this.weights[i] = Math.random()
        }
    }

    predict(input: number[]): number {
        const weighted = this.weightInput(input)
        return sigmoid(weighted)
    }

    train(input: number[], expected: number) {
        const out = this.predict(input)
        const error = out - expected
        const backpropagatedOutput = sigmoidDerivate(this.weightInput(input))

        for (let i = 0; i <= this.size; ++i) {
            if (i === this.size) {
                const delta = -this.learningRate * backpropagatedOutput * error
                this.bias += delta
            } else {
                const delta = -this.learningRate * input[i] * backpropagatedOutput * error
                this.weights[i] += delta
            }
        }
    }

    private weightInput(input: number[]): number {
        return this.bias + dot(input, this.weights)
    }

}