import { add, compositionDependencies, dotMultiply, multiply, subtract, transpose } from "mathjs"
import Neuron from "./Neuron"
import dot from "./shared/dot"
import sigmoid from "./shared/sigmoid"
import sigmoidDerivate from "./shared/sigmoidDerivate"

export type Layer = Neuron[]

export default class NeuralNetwork {

    private layers: Layer[]

    private readonly learningRate: number = 0.1

    constructor() {
        this.layers = [[new Neuron(2, [-0.16595599, -0.99977125, -0.70648822]), new Neuron(2, [0.44064899, -0.39533485, -0.81532281])], [new Neuron(2, [-0.62747958, -0.30887855, -0.20646505])]]
    }

    feedforward(input: number[]): number[] {
        return this.feed(input, 0)
    }

    train(input: number[], target: number[]) {
        const outputs = [input, this.feed(input, 0, 0), [this.feed(input, 0, 0).reduce((p, c, i) => p + (c * this.layers[1][0].weights[i])) + this.layers[1][0].bias]]
        const inputLayerOutput = outputs[0]
        const hiddenLayerOutput = outputs[1]
        const outputLayerOutput = outputs[2]

        console.log(outputs)

        // const outputError = subtract(target, outputLayerOutput)
        // const outputDelta = dotMultiply(outputError, outputLayerOutput.map(v => sigmoidDerivate(v)))
        // const hiddenError = multiply(outputDelta, transpose(this.layers[1][0].weights))
        // const hiddenDelta = dotMultiply(hiddenError, hiddenLayerOutput.map(v => sigmoidDerivate(v)))

        // this.layers[1][0].weights = add(this.layers[1][0].weights, multiply(transpose(hiddenLayerOutput), multiply(outputDelta, this.learningRate))) as number[]

        // console.log(add(this.layers[1][0].weights, multiply(transpose(hiddenLayerOutput), multiply(outputDelta, this.learningRate))))

        const error = outputLayerOutput[0] - target[0]
        const hiddenError = hiddenLayerOutput.map((o, i) => sigmoidDerivate(o) * (error * this.layers[1][0].weights[i]))

        const hiddenDerivate = inputLayerOutput.map((o, i) => hiddenError.map(e => e * o))
        const outputDerivate = inputLayerOutput.map((o, i) => o * error)

        // console.log(hiddenDerivate, "---", outputDerivate)

        const hiddenAvg = hiddenDerivate.map(d => d.reduce((p, c) => p + c, 0) / d.length)
        const oAvg = outputDerivate.reduce((p, c) => p + c, 0) / outputDerivate.length//.map(d => d.reduce((p, c) => p + c, 0) / d.length)

        // console.log(hiddenAvg, "---", oAvg)

        this.layers[0][0].bias += -this.learningRate * hiddenAvg[0]
        this.layers[0][1].bias += -this.learningRate * hiddenAvg[1]
        this.layers[1][0].bias += -this.learningRate * oAvg
        this.layers[0][0].weights = this.layers[0][0].weights.map((w, i) => w - this.learningRate * hiddenAvg[0])
        this.layers[0][1].weights = this.layers[0][1].weights.map((w, i) => w - this.learningRate * hiddenAvg[1])
        this.layers[1][0].weights = this.layers[1][0].weights.map((w, i) => w - this.learningRate * oAvg)


        // const outputs = this.feedforward(input)
        // const costFunction = (i: number) => outputs[i] - target[i]
        // const errors = outputs.map((_output, index) => costFunction(index))
        // const inputs = [input]
        // const outputLayerIdx = this.layers.length - 1

        // for (let i = 0; i < outputLayerIdx; ++i) {
        //     const last = inputs[inputs.length - 1]
        //     inputs.push(this.feed(last, 0, i))
        // }

        // for (const neuron of this.layers[outputLayerIdx]) {
        //     for (let w = 0; w < neuron.weights.length; ++w) {
        //         neuron.weights[w] += -this.learningRate * inputs[outputLayerIdx][w] * sigmoidDerivate(outputs[0 /** Index of neuron */]) * errors[0 /** Index of neuron */]
        //     }

        //     neuron.bias += -this.learningRate * sigmoidDerivate(outputs[0 /** Index of neuron */]) * errors[0 /** Index of neuron */]
        // }

        // for (let layerIdx = 0; layerIdx < outputLayerIdx; ++layerIdx) {
        //     this.layers[layerIdx].forEach((neuron, index) => {
        //         for (let w = 0; w < neuron.weights.length; ++w) {
        //             neuron.weights[w] += -this.learningRate * inputs[layerIdx][w] * sigmoidDerivate(inputs[layerIdx + 1][w]) * ((sigmoidDerivate(outputs[0]) * errors[0]) * this.layers[outputLayerIdx][0].weights[w])
        //         }

        //         neuron.bias += -this.learningRate * ((sigmoidDerivate(outputs[0]) * errors[0]) * this.layers[outputLayerIdx][0].bias)
        //     })
        // }

        // for (let layerIdx = this.layers.length - 1; layerIdx >= 0; --layerIdx) {
        //     this.layers[layerIdx].forEach((neuron, neuronIdx) => {
        //         if (layerIdx === this.layers.length - 1) {
        //             for (let weightIdx = 0; weightIdx <= neuron.weights.length; ++weightIdx) {
        //                 if (weightIdx === neuron.weights.length) {
        //                     neuron.bias += -this.learningRate * errors[neuronIdx]
        //                 } else {
        //                     neuron.weights[weightIdx] += -this.learningRate * inputs[layerIdx][weightIdx] * errors[neuronIdx]
        //                 }
        //             }
        //         } else {
        //             let errorFunc = dot(this.layers[layerIdx + 1][0].weights, inputs[layerIdx])
        //             for (let weightIdx = 0; weightIdx <= neuron.weights.length; ++weightIdx) {
        //                 if (weightIdx === neuron.weights.length) {
        //                     neuron.bias += -this.learningRate * sigmoidDerivate(errorFunc)
        //                 } else {
        //                     neuron.weights[weightIdx] += -this.learningRate * inputs[layerIdx][weightIdx] * sigmoidDerivate(errorFunc)
        //                 }
        //             }
        //         }
        //     })
        // }

        // const outputs = this.feedforward(input)
        // const hiddenOutput = this.layers[0].map(neuron => neuron.predict(input))
        // const outErrors = outputs.map((output, index) => sigmoidDerivate(output) * (output - target[index]))
        // const hiddenErrors = hiddenOutput.map((hiddenOutput, index) => sigmoidDerivate(hiddenOutput) * dot([outErrors[0], outErrors[0]], this.layers[1][0].weights))

        // this.layers[1].forEach((neuron, index) => {
        //     for (let w = 0; w <= neuron.weights.length; ++w) {
        //         if (w === neuron.weights.length) {
        //             neuron.bias += -this.learningRate * outErrors[index]
        //         } else {
        //             neuron.weights[w] += -this.learningRate * hiddenOutput[w] * outErrors[index]
        //         }
        //     }
        // })

        // this.layers[0].forEach((neuron, index) => {
        //     for (let w = 0; w <= neuron.weights.length; ++w) {
        //         if (w === neuron.weights.length) {
        //             neuron.bias += -this.learningRate * hiddenErrors[index]
        //         } else {
        //             neuron.weights[w] += -this.learningRate * input[w] * hiddenErrors[index]
        //         }
        //     }
        // })
    }

    private feed(input: number[], layerIndex: number, end?: number): number[] {
        const output: number[] = []

        for (const neuron of this.layers[layerIndex]) {
            output.push(neuron.predict(input))
        }

        if (layerIndex === (end !== undefined ? end : this.layers.length - 1)) {
            return output
        } else {
            return this.feed(output, layerIndex + 1)
        }
    }

}