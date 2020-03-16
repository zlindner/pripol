const merge = require('webpack-merge');
const { resolve } = require('path');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const base = require('./base');

module.exports = merge(base, {
    mode: 'production',
    entry: './index.tsx',
    output: {
        filename: 'js/bundle.[hash].min.js',
        path: resolve(__dirname, '../../dist'),
        publicPath: '/'
    },
    devtool: 'source-map',
    plugins: [new CleanWebpackPlugin()]
});
