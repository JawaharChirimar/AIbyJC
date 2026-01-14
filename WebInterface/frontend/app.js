// API base URL
const API_BASE_URL = 'http://localhost:5001/api';

// DOM elements
const modelSelect = document.getElementById('modelSelect');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const classifyBtn = document.getElementById('classifyBtn');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const resultsContainer = document.getElementById('resultsContainer');

// State
let selectedFile = null;

// Initialize: Load models on page load
window.addEventListener('DOMContentLoaded', () => {
    loadModels();
    
    // File input change handler
    fileInput.addEventListener('change', handleFileSelect);
    
    // Classify button click handler
    classifyBtn.addEventListener('click', handleClassify);
});

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/models`);
        const data = await response.json();
        
        if (data.error) {
            showError(`Failed to load models: ${data.error}`);
            return;
        }
        
        // Populate model select
        modelSelect.innerHTML = '<option value="">Select a model...</option>';
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.path;
            option.textContent = `${model.name} (${model.run})`;
            modelSelect.appendChild(option);
        });
        
        modelSelect.disabled = false;
        updateClassifyButtonState();
    } catch (error) {
        showError(`Failed to load models: ${error.message}`);
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    
    if (!file) {
        selectedFile = null;
        fileInfo.textContent = '';
        updateClassifyButtonState();
        return;
    }
    
    // Check file type
    if (!file.type.match(/^image\/(jpeg|jpg|png)$/)) {
        showError('Please select a JPEG or PNG image file.');
        fileInput.value = '';
        selectedFile = null;
        fileInfo.textContent = '';
        updateClassifyButtonState();
        return;
    }
    
    // Check file size (2MB limit)
    if (file.size > 2 * 1024 * 1024) {
        showError('File size exceeds 2MB limit. Please select a smaller file.');
        fileInput.value = '';
        selectedFile = null;
        fileInfo.textContent = '';
        updateClassifyButtonState();
        return;
    }
    
    selectedFile = file;
    fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    updateClassifyButtonState();
}

function updateClassifyButtonState() {
    const hasModel = modelSelect.value !== '';
    const hasFile = selectedFile !== null;
    classifyBtn.disabled = !(hasModel && hasFile);
}

async function handleClassify() {
    if (!selectedFile || !modelSelect.value) {
        return;
    }
    
    // Disable button and show loading
    classifyBtn.disabled = true;
    classifyBtn.textContent = 'Processing...';
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div>Processing image...</div>';
    resultsSection.style.display = 'block';
    
    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model_path', modelSelect.value);
        
        // Send request
        const response = await fetch(`${API_BASE_URL}/classify`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        if (data.results) {
            displayResults(data.results);
        } else {
            showError('Unexpected response format');
        }
    } catch (error) {
        showError(`Classification failed: ${error.message}`);
    } finally {
        classifyBtn.disabled = false;
        classifyBtn.textContent = 'Classify Digits';
        updateClassifyButtonState();
    }
}

function displayResults(results) {
    resultsContainer.innerHTML = '';
    
    if (results.length === 0) {
        resultsContainer.innerHTML = '<div class="loading">No digits found in the image.</div>';
        return;
    }
    
    results.forEach((result, index) => {
        const item = document.createElement('div');
        item.className = 'result-item';
        
        const img = document.createElement('img');
        img.src = `data:image/jpeg;base64,${result.image}`;
        img.alt = `Digit ${result.digit}`;
        
        const digitDiv = document.createElement('div');
        digitDiv.className = 'digit';
        digitDiv.textContent = result.digit;
        
        const confidenceDiv = document.createElement('div');
        confidenceDiv.className = 'confidence';
        confidenceDiv.innerHTML = `Confidence: <span class="confidence-value">${(result.confidence * 100).toFixed(1)}%</span>`;
        
        item.appendChild(img);
        item.appendChild(digitDiv);
        item.appendChild(confidenceDiv);
        
        resultsContainer.appendChild(item);
    });
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    resultsSection.style.display = 'block';
    resultsContainer.innerHTML = '';
}

function formatFileSize(bytes) {
    if (bytes < 1024) {
        return bytes + ' B';
    } else if (bytes < 1024 * 1024) {
        return (bytes / 1024).toFixed(1) + ' KB';
    } else {
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
}
