//
//  DigitClassifier.swift
//  AIbyJCApp
//
//  Created by Jaanvi Chirimar on 1/11/26.
//

import UIKit

struct ServerDigitResult: Codable, Sendable {
    let image: String  // Base64
    let digit: Int
    let confidence: Double
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

class DigitClassifierService {
    
    static let shared = DigitClassifierService()
    
    // ⚠️ UPDATE THIS to your Mac's IP address!

    private let serverURL = "http://192.168.1.172:5001"
    
    // Hard-code your model path (simpler than auto-loading)
    private let modelPath = "/Users/jaanvichirimar/Development/AIbyJC/DigitNN/data/modelForDE/run_2026_01_06_16_52_06/digit_classifier_epoch_03.h5"
    
    private init() {
        print("✅ Server: \(serverURL)")
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
