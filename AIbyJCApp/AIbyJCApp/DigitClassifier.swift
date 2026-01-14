//
//  DigitClassifier.swift
//  AIbyJCApp
//
//  Created by Jaanvi Chirimar on 1/11/26.
//

import UIKit
internal import Combine

struct ServerDigitResult: Codable, Sendable {
    let image: String  // Base64
    let digit: Int
    let confidence: Double
}

struct ModelInfo: Codable, Sendable, Identifiable {
    let path: String
    let name: String
    let run: String
    
    var id: String { path }
    
    var displayName: String {
        // Use the "run" name for display (e.g., "run_all_data_aug2x_4conv_64dense")
        return run != "unknown" ? run : name
    }
}

struct ModelsResponse: Codable, Sendable {
    let models: [ModelInfo]?
    let error: String?
}

struct ServerResponse: Sendable {
    let results: [ServerDigitResult]?
    let error: String?
}

extension ServerResponse: Codable {
    enum CodingKeys: String, CodingKey {
        case results
        case error
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        results = try container.decodeIfPresent([ServerDigitResult].self, forKey: .results)
        error = try container.decodeIfPresent(String.self, forKey: .error)
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(results, forKey: .results)
        try container.encodeIfPresent(error, forKey: .error)
    }
}

class DigitClassifierService: ObservableObject {
    
    static let shared = DigitClassifierService()
    
    // ⚠️ UPDATE THIS to your Mac's IP address!
    // Use "localhost" for Simulator, or "192.168.1.172" for real device
    #if targetEnvironment(simulator)
    private let serverURL = "http://localhost:5001"
    #else
    private let serverURL = "http://192.168.1.172:5001"
    #endif
    
    // Selected model path (loaded from UserDefaults or set by user)
    @Published private(set) var selectedModelPath: String?
    
    // Available models from server
    @Published var availableModels: [ModelInfo] = []
    
    // Connection status
    @Published var isServerReachable: Bool?
    
    private init() {
        print("✅ Server: \(serverURL)")
        loadSelectedModel()
    }
    
    // MARK: - Model Management
    
    private func loadSelectedModel() {
        selectedModelPath = UserDefaults.standard.string(forKey: "selectedModelPath")
        if let path = selectedModelPath {
            print("📦 Loaded saved model: \(path)")
        }
    }
    
    func selectModel(_ model: ModelInfo) {
        selectedModelPath = model.path
        UserDefaults.standard.set(model.path, forKey: "selectedModelPath")
        print("✅ Selected model: \(model.displayName)")
    }
    
    func fetchAvailableModels(completion: @escaping (Result<[ModelInfo], Error>) -> Void) {
        guard let url = URL(string: "\(serverURL)/api/models") else {
            completion(.failure(NSError(domain: "URLError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        print("🔍 Fetching models from: \(url.absoluteString)")
        
        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            // Check HTTP response first
            if let httpResponse = response as? HTTPURLResponse {
                print("📥 Models API HTTP Status: \(httpResponse.statusCode)")
            }
            
            if let error = error {
                print("❌ Failed to fetch models: \(error.localizedDescription)")
                print("   Error code: \((error as NSError).code)")
                print("   Error domain: \((error as NSError).domain)")
                completion(.failure(error))
                return
            }
            
            guard let data = data else {
                print("❌ No data received from models API")
                completion(.failure(NSError(domain: "NoData", code: -1, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                return
            }
            
            print("📥 Received \(data.count) bytes from models API")
            
            // Print raw response for debugging
            if let rawResponse = String(data: data, encoding: .utf8) {
                print("📄 Raw models response: \(rawResponse)")
            }
            
            do {
                let decoder = JSONDecoder()
                let response = try decoder.decode(ModelsResponse.self, from: data)
                
                if let error = response.error {
                    print("❌ Server error: \(error)")
                    completion(.failure(NSError(domain: "ServerError", code: -1, userInfo: [NSLocalizedDescriptionKey: error])))
                    return
                }
                
                if let models = response.models {
                    print("✅ Found \(models.count) models")
                    for model in models {
                        print("  📦 \(model.displayName) - \(model.name)")
                        print("      Path: \(model.path)")
                    }
                    
                    DispatchQueue.main.async {
                        self.availableModels = models
                        
                        // If no model selected yet, select the first one
                        if self.selectedModelPath == nil, let firstModel = models.first {
                            self.selectModel(firstModel)
                        }
                    }
                    
                    completion(.success(models))
                } else {
                    print("❌ No models in response")
                    completion(.failure(NSError(domain: "NoModels", code: -1, userInfo: [NSLocalizedDescriptionKey: "No models found"])))
                }
                
            } catch {
                print("❌ JSON decode error: \(error)")
                if let decodingError = error as? DecodingError {
                    switch decodingError {
                    case .keyNotFound(let key, let context):
                        print("   Missing key: \(key.stringValue)")
                        print("   Context: \(context.debugDescription)")
                    case .typeMismatch(let type, let context):
                        print("   Type mismatch: expected \(type)")
                        print("   Context: \(context.debugDescription)")
                    case .valueNotFound(let type, let context):
                        print("   Value not found: \(type)")
                        print("   Context: \(context.debugDescription)")
                    case .dataCorrupted(let context):
                        print("   Data corrupted: \(context.debugDescription)")
                    @unknown default:
                        print("   Unknown decoding error")
                    }
                }
                completion(.failure(error))
            }
        }.resume()
    }
    
    // Test server connection
    func testConnection(completion: @escaping (Result<String, Error>) -> Void) {
        guard let url = URL(string: "\(serverURL)/api/health") else {
            completion(.failure(NSError(domain: "URLError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        print("🔍 Testing connection to: \(url.absoluteString)")
        
        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("❌ Connection test failed: \(error.localizedDescription)")
                completion(.failure(error))
                return
            }
            
            if let httpResponse = response as? HTTPURLResponse {
                print("✅ Server responded with status: \(httpResponse.statusCode)")
                
                if httpResponse.statusCode == 200 {
                    completion(.success("Server is reachable (Status: \(httpResponse.statusCode))"))
                } else {
                    completion(.failure(NSError(domain: "ServerError", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "Server returned status \(httpResponse.statusCode)"])))
                }
            } else {
                completion(.failure(NSError(domain: "ServerError", code: -1, userInfo: [NSLocalizedDescriptionKey: "No HTTP response"])))
            }
        }.resume()
    }
    
    
    func processImage(_ image: UIImage, completion: @escaping (Result<[ServerDigitResult], Error>) -> Void) {
        
        guard let modelPath = selectedModelPath else {
            print("❌ No model selected")
            completion(.failure(NSError(domain: "NoModel", code: -1, userInfo: [NSLocalizedDescriptionKey: "No model selected. Please select a model first."])))
            return
        }
        
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            print("❌ Failed to convert image to JPEG data")
            completion(.failure(NSError(domain: "ImageError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Could not convert image to JPEG"])))
            return
        }
        
        print("📊 Image size: \(imageData.count) bytes")
        
        // Create multipart form data
        let boundary = UUID().uuidString
        var body = Data()
        
        // Add model_path
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"model_path\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(modelPath)\r\n".data(using: .utf8)!)
        
        // Add file
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        // Create request
        guard let url = URL(string: "\(serverURL)/api/classify") else {
            print("❌ Invalid URL: \(serverURL)/api/classify")
            completion(.failure(NSError(domain: "URLError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid server URL"])))
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        request.timeoutInterval = 30
        
        print("📤 Sending to server: \(url.absoluteString)")
        print("📦 Request size: \(body.count) bytes")
        
        // Send request
        URLSession.shared.dataTask(with: request) { data, response, error in
            // Check HTTP response
            if let httpResponse = response as? HTTPURLResponse {
                print("📥 HTTP Status: \(httpResponse.statusCode)")
            }
            
            if let error = error {
                print("❌ Network Error: \(error.localizedDescription)")
                completion(.failure(error))
                return
            }
            
            guard let data = data else {
                print("❌ No data received from server")
                completion(.failure(NSError(domain: "NoData", code: -1, userInfo: [NSLocalizedDescriptionKey: "No data received from server"])))
                return
            }
            
            print("📥 Received \(data.count) bytes from server")
            
            // Print raw response for debugging
            if let rawResponse = String(data: data, encoding: .utf8) {
                print("📄 Raw response: \(rawResponse)")
            }
            
            do {
                let decoder = JSONDecoder()
                let serverResponse = try decoder.decode(ServerResponse.self, from: data)
                
                if let error = serverResponse.error {
                    print("❌ Server returned error: \(error)")
                    completion(.failure(NSError(domain: "ServerError", code: -1, userInfo: [NSLocalizedDescriptionKey: error])))
                    return
                }
                
                if let results = serverResponse.results {
                    print("✅ Successfully decoded \(results.count) digits")
                    
                    // Verify base64 images are valid
                    for (index, result) in results.enumerated() {
                        if let imageData = Data(base64Encoded: result.image) {
                            print("  ✓ Digit \(index): \(result.digit) (\(String(format: "%.1f%%", result.confidence * 100))) - image size: \(imageData.count) bytes")
                        } else {
                            print("  ⚠️ Digit \(index): Failed to decode base64 image")
                        }
                    }
                    
                    completion(.success(results))
                } else {
                    print("❌ No results in server response")
                    completion(.failure(NSError(domain: "NoResults", code: -1, userInfo: [NSLocalizedDescriptionKey: "Server returned no results"])))
                }
                
            } catch {
                print("❌ JSON Decode error: \(error)")
                completion(.failure(error))
            }
        }.resume()
    }
}
