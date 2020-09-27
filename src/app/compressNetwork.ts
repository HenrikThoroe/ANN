import Network, { NetworkJSON, NetworkState } from "../nets/Network";

export default function compressNetwork(network: Network, training: NetworkState[]): NetworkJSON {
    return {
        meta: network.getInfo(),
        training: training
    }
}