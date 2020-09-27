import Network, { NetworkJSON, NetworkState, TrainingBundle } from "../nets/Network";

export default function parseNetwork(json: NetworkJSON): TrainingBundle {
    return {
        training: json.training,
        network: new Network(json.meta.dimensions, json.meta.activationName, json.meta.name)
    }
}