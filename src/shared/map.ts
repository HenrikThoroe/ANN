export default function map<T>(num: number, action: (num: number) => T): T[] {
    const out: T[] = []
    for (let i = 0; i < num; ++i) {
        out.push(action(i))
    }
    return out
}