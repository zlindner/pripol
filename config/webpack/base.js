const { resolve } = require('path');
const { CheckerPlugin } = require('awesome-typescript-loader');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
    resolve: {
        extensions: ['.ts', '.tsx', '.js', '.jsx']
    },
    context: resolve(__dirname, '../../client/src'),
    module: {
        rules: [
            {
                test: /\.js$/,
                use: ['babel-loader', 'source-map-loader'],
                exclude: /node_modules/
            },
            {
                test: /\.tsx?$/,
                use: ['babel-loader', 'awesome-typescript-loader']
            },
            {
                test: /\.svg$/,
                use: ['babel-loader', 'react-svg-loader']
            }
        ]
    },
    plugins: [new CheckerPlugin(), new HtmlWebpackPlugin({ template: 'index.html' })]
};
