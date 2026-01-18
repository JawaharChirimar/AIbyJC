const path = require("path");

const { CleanWebpackPlugin } = require("clean-webpack-plugin");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const MiniCssExtractPlugin = require("mini-css-extract-plugin");

const mode = (process.env.NODE_ENV === "development" ? "development" : "production");

module.exports = {
	mode: mode,
	entry: "./src/main.tsx",
	target: 'web',

	output: {
		path: path.resolve(__dirname, "dist"),
		filename: "[name].[contenthash].js"
	},

	devServer: {
		historyApiFallback: true,
		port: 5002,
		proxy: {
			'/api': {
				target: 'http://localhost:5001',
				changeOrigin: true
			}
		}
	},

	module: {
		rules: [
			{
				test: /\.(ts|tsx)?$/,
				loader: "ts-loader"
			},
			{
				test: /\.css$/i,
				use: [
					MiniCssExtractPlugin.loader,
					"css-loader",
				],
			},
			{
				test: /\.scss$/i,
				use: [
					MiniCssExtractPlugin.loader,
					"css-loader",
					{
						loader: "sass-loader",
						options: {
							api: "modern-compiler"
						}
					}
				],
			},
		],
	},

	plugins: [
		new CleanWebpackPlugin(),
		new MiniCssExtractPlugin({
			filename: "[name].[contenthash].css",
			chunkFilename: "[id].css",
			ignoreOrder: false,
		}),

		new HtmlWebpackPlugin({
			title: "Digit Classifier",
			template: "./src/index.html",
			publicPath: "/"
		})
	],

	resolve: {
		extensions: [".ts", ".tsx", ".js", ".json"],
		alias: {
			"react": "@preact/compat",
			"react-dom": "@preact/compat"
		},
		modules: [
			path.resolve("./src"),
			path.resolve("./node_modules"),
			path.resolve("./../node_modules"),
			path.resolve("./../../node_modules")
		]
	}
};