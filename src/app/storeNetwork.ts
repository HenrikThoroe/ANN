import Network, { NetworkState } from "../nets/Network";
import mongo from "mongodb"

export default async function storeNetwork(network: Network, trainingData: NetworkState[], database: mongo.Db): Promise<void> {
    return new Promise((res, rej) => {
        const collection = database.collection("networks")
        const json = {
            meta: network.getInfo(),
            training: trainingData
        }

        collection.insertOne(json, (err, r) => {
            if (err) {
                rej()
            } else {
                res()
            }
        })
    })
}