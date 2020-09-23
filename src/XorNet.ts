import sigmoid from "./shared/sigmoid"
import sigmoidDerivate from "./shared/sigmoidDerivate"

export default class XorNet {
    /** The number of neurons in each layer. Input layer exclusive. */
    private readonly neurons = [2, 1]

    /** The values for the bias of each neuron */
    private biases = [[-0.16595599, 0.44064899], [-0.62747958]]

    /** The values for the weights of each neuron */
    private weights = [[[-0.99977125, -0.70648822], [-0.39533485, -0.81532281]], [[-0.30887855, -0.20646505]]]

    /** The training data set. It contains every possible input with the target output. */
    private readonly data: [number[], number][] = [
        [[1, 0], 1],
        [[0, 1], 1],
        [[0, 0], 0],
        [[1, 1], 0]
    ]

    /** The summed error of the last training session */
    private error = 0

    /** The number of iterations of the last training session */
    private trainingCycles = 0

    /** Predict the XOR value of two bits */
    predict(...value: [number, number]): number {
        return this.feedforward(value)[2][0]
    }

    /** Trains the network using backpropagation */
    train(epochs: number = 1000000) {
        this.error = 0
        this.trainingCycles = epochs * this.data.length

        for (let i = 0; i < epochs; ++i) {
            this.data.forEach(d => {
                this.error += this.backpropagate(d[0], [d[1]])
            })
        }
    }

    /** 
     * Returns the error of the last training session. 
     * If none was performed a new one will be run first with one epoch to keep training success low and not changing the error much. 
     */
    getError() {
        if (this.trainingCycles === 0) {
            this.train(1)
        }

        return this.error / this.trainingCycles
    }

    /** Triggers a neuron in the specified layer
     * @returns A tuple with the function value (weights * input + bias) and the activation (sigmoid)
     */
    private trigger(layer: number, neuron: number, data: number[]) {
        const w: number[] = this.weights[layer][neuron]
        const b: number = this.biases[layer][neuron]
        const z = data.map((d, i) => d * w[i]).reduce((p, c) => p + c) + b
        const a = sigmoid(z)
    
        return [z, a]
    }
    
    /** Feeds the data through the newtwork.
     * @returns The output of each layer including the input layer
     */
    private feedforward(data: number[]): number[][] {
        const result: number[][] = [data]
    
        for (let l = 0; l < 2; ++l) {
            const out: number[] = []
    
            for (let i = 0; i < this.neurons[l]; ++i) {
                out.push(this.trigger(l, i, data)[1])
            }
    
            result.push(out)
            data = [...out]
        }
    
        return result
    }
    
    private backpropagate(data: number[], target: number[]) {
        // Get the output of each layer by propagating the data forwards through the network
        const [input_layer_output, hidden_layer_output, output_layer_output] = this.feedforward(data)
    
        // The error of the output neuron => distance of the real output to the desired output squared
        // The square removes negative values, results in a function with minimum and is easy to derive. 
        const error = Math.pow(output_layer_output[0] - target[0], 2)

        // A delta value for each output weight. 
        // It indicates in which direction and how much the weight should be adjusted in order to minimise the error.
        // It calculates by going the network backwards up to the neuron which is connected to the weight:
        // Because: error = (out - target)^2 = (sigmoid(function) - target)^2 = (sigmoid(weights * inputs + bias) - target)^2
        // The derivation (path which leads to a minimum / maximum) is:
        //      error' = 2 * (out - target) * (out - target)' = 2 * (out - target) * sigmoid'(function) * function' = 2 * (out - target) * sigmoid'(function) * inputs
        // Note: Because inputs is an array (vector) error' / delta is an array too. It is obvious because each weight needs to be adjusted differently
        const outputDeltas = hidden_layer_output.map(a => (2 * (output_layer_output[0] - target[0])) * sigmoidDerivate(this.trigger(1, 0, hidden_layer_output)[1]) * a)

        // The delta for the bias calculates similiar to the delta of the weights.
        // The difference is that the derivation of "function" is with respect to bias and not to weights.
        // So it is not inputs but 1
        const outputBiasDelta = (2 * (output_layer_output[0] - target[0])) * sigmoidDerivate(this.trigger(1, 0, hidden_layer_output)[1])
        
        // The hidden layer calculates it's weights similiar to the output layer but with the error being the delta of the connected weight of the output layer.
        // Because each weight of the output layer is connected to a single neuron in the hidden layer, we can iterate over the deltas and see each delta as the error of one neuron. 
        // The output delta is the hidden layer error bacause:
        //      If a weight of the output needs to be less important, the connected input (the output of one of the hidden neurons) should be less important too.
        const hiddenDeltas = outputDeltas.map((delta, n) => input_layer_output.map(x => delta * sigmoidDerivate(this.trigger(0, n, input_layer_output)[0]) * x))
        const hiddenBiasDeltas = outputDeltas.map((delta, n) => delta * sigmoidDerivate(this.trigger(0, n, input_layer_output)[0]))
    
        // At last the calculated deltas have to added to the linked weights / biases.
        // The 0.1 is the learning rate. If choosen too high a minimum could be missed because the backpropagation would simply make too large steps.
        // If choosen too low the training would need unneccessary long.

        this.biases[1][0] -= 0.1 * outputBiasDelta
        this.biases[0][0] -= 0.1 * hiddenBiasDeltas[0]
        this.biases[0][1] -= 0.1 * hiddenBiasDeltas[1]
    
        this.weights[1][0] = this.weights[1][0].map((w, i) => w - 0.1 * outputDeltas[i])
        this.weights[0][0] = this.weights[0][0].map((w, i) => w - 0.1 * hiddenDeltas[0][i])
        this.weights[0][1] = this.weights[0][1].map((w, i) => w - 0.1 * hiddenDeltas[1][i])
    
        return error
    }
}