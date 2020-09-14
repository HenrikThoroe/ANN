export default function dot(x: number[], y: number[]): number {
    if (x.length !== y.length) {
        throw new Error(`Dot product requires that both inputs are of the same size. Given: ${x}[${x.length}] and ${y}[${y.length}]`)
    }

    return x.reduce((prev, current, idx) => prev + current * y[idx], 0)
}