export default function range(length: number): number[] {
    const out: number[] = []
    for (let i = 0; i < length; ++i) {
        out.push(i)
    }
    return out
}