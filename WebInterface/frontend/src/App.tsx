import { h, Component } from "preact";
import React from "react";
import Select from "react-select";
import "App.scss";

const API_BASE_URL = 'http://localhost:5001/api';

interface Model {
	path: string;
	name: string;
	run: string;
}

interface ClassificationResult {
	image: string;
	digit: number;
	confidence: number;
}

interface ModelsByDirectory {
	[key: string]: Model[];
}

interface AppState {
	models: ModelsByDirectory;
	selectedModel: string;
	selectedFile: File | null;
	results: ClassificationResult[];
	error: string;
	loading: boolean;
	modelsLoading: boolean;
	showResults: boolean;
}

export class App extends Component<{}, AppState> {
	constructor() {
		super();
		this.state = {
			models: {},
			selectedModel: "",
			selectedFile: null,
			results: [],
			error: "",
			loading: false,
			modelsLoading: true,
			showResults: false
		};
	}

	componentDidMount() {
		this.loadModels();
	}

	loadModels = async () => {
		try {
			const response = await fetch(`${API_BASE_URL}/models`);
			const data = await response.json();
			
			if (data.error) {
				this.setState({ error: `Failed to load models: ${data.error}` });
				return;
			}
			
			const modelsByDirectory: ModelsByDirectory = {};
			data.models.forEach((model: Model) => {
				const dir = model.run || 'unknown';
				if (!modelsByDirectory[dir]) {
					modelsByDirectory[dir] = [];
				}
				modelsByDirectory[dir].push(model);
			});
			
			this.setState({ models: modelsByDirectory, modelsLoading: false });
		} catch (err: any) {
			this.setState({ error: `Failed to load models: ${err.message}`, modelsLoading: false });
		}
	};

	handleFileSelect = (event: Event) => {
		const target = event.target as HTMLInputElement;
		const file = target.files?.[0];
		
		if (!file) {
			this.setState({ selectedFile: null });
			return;
		}
		
		if (!file.type.match(/^image\/(jpeg|jpg|png)$/)) {
			alert('Please select a JPEG or PNG image file.');
			target.value = '';
			this.setState({ selectedFile: null });
			return;
		}
		
		if (file.size > 2 * 1024 * 1024) {
			alert('File size exceeds 2MB limit. Please select a smaller file.');
			target.value = '';
			this.setState({ selectedFile: null });
			return;
		}
		
		this.setState({ selectedFile: file, error: "", showResults: false });
	};

	handleModelChange = (value: string) => {
		this.setState({ 
			selectedModel: value, 
			showResults: false, 
			error: "", 
			results: [] 
		});
	};

	handleClassify = async () => {
		const { selectedFile, selectedModel } = this.state;
		if (!selectedFile || !selectedModel) {
			return;
		}
		
		this.setState({ loading: true, showResults: true, error: "", results: [] });
		
		try {
			const formData = new FormData();
			formData.append('file', selectedFile);
			formData.append('model_path', selectedModel);
			
			const response = await fetch(`${API_BASE_URL}/classify`, {
				method: 'POST',
				body: formData
			});
			
			const data = await response.json();
			
			if (data.error) {
				this.setState({ error: data.error });
				return;
			}
			
			if (data.results) {
				this.setState({ results: data.results });
			} else {
				this.setState({ error: 'Unexpected response format' });
			}
		} catch (err: any) {
			this.setState({ error: `Classification failed: ${err.message}` });
		} finally {
			this.setState({ loading: false });
		}
	};

	render() {
		const { models, selectedModel, selectedFile, results, error, loading, modelsLoading, showResults } = this.state;
		const canClassify = selectedModel !== "" && selectedFile !== null;

		// Convert models to react-select format with groups
		const groupedOptions = Object.keys(models).sort().map(dir => ({
			label: dir,
			options: models[dir].map(model => ({
				value: model.path,
				label: model.name,
				dir: dir
			}))
		}));

		const selectedOption = groupedOptions
			.flatMap(group => group.options)
			.find(opt => opt.value === selectedModel) || null;

		return (
			<div class="container">
				<main>
					<header>
						<h1>Digit Classifier</h1>
						<p>Use Deep Learning to classify handwritten digits in uploaded images.</p>
					</header>
					

					<div class="upload-section">
						<div class="form-group">
							<Select
								value={selectedOption}
								options={groupedOptions}
								onChange={(option: { value: string; label: string; dir?: string } | null) => {
									this.handleModelChange(option ? option.value : "");
								}}
								formatOptionLabel={(option: { value: string; label: string; dir?: string }, { context }) => {
									// Show "Dir: Model" when selected, just "Model" in dropdown
									if (context === 'value' && option.dir) {
										return `${option.dir}: ${option.label}`;
									}
									return option.label;
								}}
								isDisabled={modelsLoading}
								placeholder={modelsLoading ? "Loading models..." : "Select a model..."}
								className="react-select-container"
								classNamePrefix="react-select"
							/>
						</div>

						<div class="form-group">
							<input 
								type="file" 
								id="fileInput" 
								accept="image/jpeg,image/jpg,image/png"
								onChange={this.handleFileSelect}
							/>
							<div class="file-info" id="fileInfo"></div>
							<small>Image must be jpeg/png file under 2MB.</small>
						</div>

						<button 
							id="classifyBtn" 
							class="btn-primary" 
							disabled={!canClassify || loading}
							onClick={this.handleClassify}
						>
							{loading ? "Processing..." : "CLASSIFY DIGITS"}
						</button>
					</div>

					{showResults && (
						<div class="results-section">
							<h2>Results</h2>
							{error && (
								<div id="errorMessage" class="error-message">
									{error}
								</div>
							)}
							<div id="resultsContainer" class="results-container">
								{loading && (
									<div class="loading">
										<div class="spinner"></div>
										Processing image...
									</div>
								)}
								{!loading && results.length === 0 && !error && (
									<div class="loading">No digits found in the image.</div>
								)}
								{results.map((result, index) => (
									<div 
										class={`result-item ${result.confidence < 0.6666 ? 'low-confidence' : ''}`}
										key={index}
									>
										<img 
											src={`data:image/jpeg;base64,${result.image}`} 
											alt={`Digit ${result.digit}`}
										/>
										<div class="digit">{result.digit}</div>
										<div class="confidence">
											Confidence: <span class="confidence-value">{(result.confidence * 100).toFixed(1)}%</span>
										</div>
									</div>
								))}
							</div>
						</div>
					)}
				</main>
			</div>
		);
	}
}