import { Db } from "mongodb";
import { NetworkJSON, TrainingBundle } from "../nets/Network";
import parseNetwork from "./parseNetwork";

export default async function loadNetworks(db: Db): Promise<TrainingBundle[]> {
    return new Promise((res, rej) => {
        const nets: TrainingBundle[] = []

        db.collection("networks").find().forEach((doc) => {
            nets.push(parseNetwork(doc as NetworkJSON))
        })  

        res(nets)
    })
} 