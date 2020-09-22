import { e } from "mathjs"
import dot from "./shared/dot"
import sigmoid from "./shared/sigmoid"
import sigmoidDerivate from "./shared/sigmoidDerivate"

export default class Neuron {

    private readonly learningRate: number = 0.05

    private readonly size: Number

    public bias: number

    public weights: number[]

    constructor(size: number, weights?: number[]) {
        this.size = size
        this.bias = 1
        this.weights = new Array(size)

        if (weights) {
            this.bias = weights[0]
            this.weights = weights.slice(1)
        } else {
            for (let i = 0; i < size; ++i) {
                this.weights[i] = Math.random()
            }
        }
    }

    predict(input: number[]): number {
        return this.activation(this.func(input))
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

    func(x: number[]): number {
        return this.bias + dot(x, this.weights)
    } 

    activation(x: number): number {
        return sigmoid(x)
    }

    private weightInput(input: number[]): number {
        return this.bias + dot(input, this.weights)
    }

}