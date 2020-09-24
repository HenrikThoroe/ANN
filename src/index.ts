import XorNet from "./nets/XorNet";

const xor =  new XorNet()

console.log(`Error before training: ${xor.getError()}`)

xor.train()

console.log(`Error after training: ${xor.getError()}`)
console.log(`Prediction for [0, 1]: ${xor.predict(0, 1)}`)
console.log(`Prediction for [1, 1]: ${xor.predict(1, 1)}`)
console.log(`Prediction for [1, 0]: ${xor.predict(1, 0)}`)
console.log(`Prediction for [0, 0]: ${xor.predict(0, 0)}`)