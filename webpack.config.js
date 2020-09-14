const path = require('path');

module.exports = {
    mode: "production",
    entry: './src/index.ts',
    target: "node",
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                use: 'ts-loader',
                exclude: /node_modules/,
            }
        ],
    },
    resolve: {
        extensions: [ '.tsx', '.ts', '.js' ],
    },
    output: {
        filename: 'ann.js',
        path: path.resolve(__dirname, 'dist'),
    },
};