const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
    resolve: {
        extensions: ['.ts', '.tsx', '.js', '.jsx']
    },
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
            }
        ]
    },
    plugins: [new HtmlWebpackPlugin()],
    externals: {
        react: 'React',
        'react-dom': 'ReactDOM'
    },
    performance: {
        hints: false
    }
};
