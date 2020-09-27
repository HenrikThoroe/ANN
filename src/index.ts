import compressNetwork from "./app/compressNetwork";
import Network, { NetworkState, TrainingBundle, TrainingSet } from "./nets/Network";
import XorNet from "./nets/XorNet";
import range from "./shared/range";
import express from "express"

const app = express()
const port = 3210
const trainedNetworks = [
    runNetwork()
]

app.get("/", (req, res) => {
    res.send("Hello, world")
})

app.get("/nets", async (req, res) => {
    res.json({ ids: range(trainedNetworks.length) })
})

app.get("/net", async (req, res) => {
    const id = parseInt(req.query.id as string)

    if (id !== undefined && typeof id === "number" && trainedNetworks.length > id) {
        res.json(compressNetwork(trainedNetworks[id].network, trainedNetworks[id].training)) 
    } else {
        res.statusCode = 404
        res.send("Not Found")
    }
})

app.listen(port, () => {
    console.log(`Server listening on port ${port}`)
})

function runXor() {
    const xor =  new XorNet()

    console.log(`Error before training: ${xor.getError()}`)

    xor.train()

    console.log(`Error after training: ${xor.getError()}`)
    console.log(`Prediction for [0, 1]: ${xor.predict(0, 1)}`)
    console.log(`Prediction for [1, 1]: ${xor.predict(1, 1)}`)
    console.log(`Prediction for [1, 0]: ${xor.predict(1, 0)}`)
    console.log(`Prediction for [0, 0]: ${xor.predict(0, 0)}`)
}

function runNetwork(): TrainingBundle {
    const net = new Network([2, 2, 1], "sigmoid", "xor")
    const history: NetworkState[] = []
    const data: TrainingSet[] = [
        { input: [0, 0], target: [0] }, 
        { input: [1, 1], target: [0] },
        { input: [1, 0], target: [1] }, 
        { input: [0, 1], target: [1] }
    ]

    console.log(net.error(data))
    net.train(1000000, data, epoch => {
        if (epoch % 100 || epoch >= 1000000 - 1) {
            history.push(net.getState())
        }
    })
    console.log(`Prediction for [${data[0].input[0]}, ${data[0].input[1]}]: ${net.predict(data[0].input)}`)
    console.log(`Prediction for [${data[1].input[0]}, ${data[1].input[1]}]: ${net.predict(data[1].input)}`)
    console.log(`Prediction for [${data[2].input[0]}, ${data[2].input[1]}]: ${net.predict(data[2].input)}`)
    console.log(`Prediction for [${data[3].input[0]}, ${data[3].input[1]}]: ${net.predict(data[3].input)}`)
    console.log(net.error(data))

    return { network: net, training: history }
}