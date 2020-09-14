import sigmoid from "./sigmoid";

export default function sigmoidDerivate(x: number): number {
    const sig = sigmoid(x)
    return sig * (1 - sig)
}