//
//  ContentView.swift
//  handwriting
//
//  Created by Jaanvi Chirimar on 1/3/26.
//

import SwiftUI
import SwiftData

@Model
class SavedPhoto {
    var originalImageData: Data
    var processedImageData: Data
    var capturedDate: Date
    var fileSizeKB: Double
    var detectedDigits: [DetectedDigitData]?
    
    init(originalImageData: Data, processedImageData: Data, capturedDate: Date, fileSizeKB: Double, detectedDigits: [DetectedDigitData]? = nil) {
        self.originalImageData = originalImageData
        self.processedImageData = processedImageData
        self.capturedDate = capturedDate
        self.fileSizeKB = fileSizeKB
        self.detectedDigits = detectedDigits
    }
}

struct DetectedDigitData: Codable {
    let lineNumber: Int
    let digitNumber: Int
    let predictedDigit: Int
    let confidence: Double
    let imageData: Data
    
    // Custom decoder to handle missing keys from old data
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        // Provide defaults for potentially missing keys
        self.lineNumber = (try? container.decode(Int.self, forKey: .lineNumber)) ?? 0
        self.digitNumber = (try? container.decode(Int.self, forKey: .digitNumber)) ?? 0
        self.predictedDigit = try container.decode(Int.self, forKey: .predictedDigit)
        self.confidence = try container.decode(Double.self, forKey: .confidence)
        self.imageData = try container.decode(Data.self, forKey: .imageData)
    }
    
    // Regular initializer for creating new instances
    init(lineNumber: Int, digitNumber: Int, predictedDigit: Int, confidence: Double, imageData: Data) {
        self.lineNumber = lineNumber
        self.digitNumber = digitNumber
        self.predictedDigit = predictedDigit
        self.confidence = confidence
        self.imageData = imageData
    }
}

struct ContentView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \SavedPhoto.capturedDate, order: .reverse) private var savedPhotos: [SavedPhoto]
    
    @State private var showCamera = false
    @State private var showPhotoPicker = false
    @State private var showAlert = false
    @State private var alertMessage = ""
    @State private var isProcessing = false
    @State private var processingStatus = ""
    @State private var showDebugMenu = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                if isProcessing {
                    VStack(spacing: 15) {
                        ProgressView()
                            .scaleEffect(1.5)
                        Text(processingStatus)
                            .font(.headline)
                    }
                    .padding(40)
                    
                } else if let photo = savedPhotos.first,
                          let image = UIImage(data: photo.originalImageData) {
                    
                    ScrollView {
                        VStack(spacing: 15) {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 300)
                                .cornerRadius(10)
                                .shadow(radius: 5)
                            
                            VStack(spacing: 5) {
                                Text("Captured: \(photo.capturedDate.formatted())")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                                
                                Text("Size: \(String(format: "%.1f", photo.fileSizeKB)) KB")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            }
                            
                            if let digits = photo.detectedDigits, !digits.isEmpty {
                                VStack(alignment: .leading, spacing: 10) {
                                    Text("Detected \(digits.count) Digits")
                                        .font(.headline)
                                        .padding(.horizontal)
                                    
                                    ForEach(Array(digits.enumerated()), id: \.offset) { _, digit in
                                        HStack(spacing: 15) {
                                            if let digitImage = UIImage(data: digit.imageData) {
                                                Image(uiImage: digitImage)
                                                    .resizable()
                                                    .scaledToFit()
                                                    .frame(width: 50, height: 50)
                                                    .background(Color.black)
                                                    .cornerRadius(5)
                                                    .overlay(
                                                        RoundedRectangle(cornerRadius: 5)
                                                            .stroke(Color.gray, lineWidth: 1)
                                                    )
                                            }
                                            
                                            VStack(alignment: .leading, spacing: 2) {
                                                Text("Line \(digit.lineNumber), Digit \(digit.digitNumber)")
                                                    .font(.caption)
                                                    .foregroundColor(.secondary)
                                            }
                                            
                                            Spacer()
                                            
                                            VStack(spacing: 2) {
                                                Text("\(digit.predictedDigit)")
                                                    .font(.system(size: 32, weight: .bold))
                                                    .foregroundColor(.blue)
                                                
                                                Text("\(String(format: "%.1f%%", digit.confidence * 100))")
                                                    .font(.caption)
                                                    .foregroundColor(.gray)
                                            }
                                        }
                                        .padding()
                                        .background(Color.gray.opacity(0.1))
                                        .cornerRadius(10)
                                    }
                                }
                                .padding(.horizontal)
                            }
                            
                            Text("\(savedPhotos.count) of 3 photos stored")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                    }
                    
                } else {
                    VStack(spacing: 20) {
                        Image(systemName: "camera.viewfinder")
                            .font(.system(size: 80))
                            .foregroundStyle(.gray)
                        
                        Text("Take a photo of handwritten DIGITS")
                            .font(.headline)
                            .foregroundColor(.gray)
                        
                        Text("The app will detect and classify each digit")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                }
                
                Spacer()
                
                HStack(spacing: 15) {
                    Button(action: { showCamera = true }) {
                        Label("Take Photo", systemImage: "camera.fill")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                    .disabled(isProcessing)
                    
                    Button(action: { showPhotoPicker = true }) {
                        Label("Choose Photo", systemImage: "photo.on.rectangle")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.green)
                            .cornerRadius(10)
                    }
                    .disabled(isProcessing)
                }
                .padding()
            }
            .navigationTitle("Digit Classifier")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showDebugMenu.toggle() }) {
                        Image(systemName: "ladybug")
                    }
                }
            }
        }
        .sheet(isPresented: $showCamera) {
            ImagePicker(sourceType: .camera) { saveImageToDatabase($0) }
        }
        .sheet(isPresented: $showPhotoPicker) {
            ImagePicker(sourceType: .photoLibrary) { saveImageToDatabase($0) }
        }
        .sheet(isPresented: $showDebugMenu) {
            NavigationView {
                List {
                    Section("Server Connection") {
                        Text("Server: http://192.168.1.172:5001")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Button(action: testServerConnection) {
                            Label("Test Connection", systemImage: "network")
                        }
                    }
                    
                    Section("Database") {
                        Text("\(savedPhotos.count) photos stored")
                        
                        if !savedPhotos.isEmpty {
                            Button(role: .destructive, action: clearAllPhotos) {
                                Label("Clear All Photos", systemImage: "trash")
                            }
                        }
                    }
                    
                    Section("Debug Info") {
                        if let photo = savedPhotos.first {
                            Text("Latest photo: \(photo.fileSizeKB, specifier: "%.1f") KB")
                            Text("Digits: \(photo.detectedDigits?.count ?? 0)")
                            Text("Original size: \(photo.originalImageData.count / 1024) KB")
                            Text("Processed size: \(photo.processedImageData.count / 1024) KB")
                        }
                    }
                }
                .navigationTitle("Debug")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button("Done") {
                            showDebugMenu = false
                        }
                    }
                }
            }
        }
        .alert("Status", isPresented: $showAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(alertMessage)
        }
        .onAppear {
            // Clean up any corrupted data on first launch
            cleanupCorruptedPhotos()
        }
    }
    
    func testServerConnection() {
        isProcessing = true
        processingStatus = "Testing connection..."
        
        DigitClassifierService.shared.testConnection { result in
            DispatchQueue.main.async {
                self.isProcessing = false
                self.processingStatus = ""
                self.showDebugMenu = false
                
                switch result {
                case .success(let message):
                    self.alertMessage = "✅ \(message)"
                case .failure(let error):
                    self.alertMessage = "❌ Connection failed:\n\(error.localizedDescription)\n\nMake sure Flask server is running and you're on the same WiFi network."
                }
                
                self.showAlert = true
            }
        }
    }
    
    func clearAllPhotos() {
        for photo in savedPhotos {
            modelContext.delete(photo)
        }
        
        do {
            try modelContext.save()
        } catch {
            print("Error clearing photos: \(error)")
        }
        
        showDebugMenu = false
    }
    
    func cleanupCorruptedPhotos() {
        // Try to access all photos to see if any are corrupted
        for photo in savedPhotos {
            // Accessing detectedDigits will trigger decoding
            _ = photo.detectedDigits
        }
    }
    
    func saveImageToDatabase(_ image: UIImage) {
        isProcessing = true
        processingStatus = "Uploading to server..."
        
        print("🖼️ Starting image processing...")
        print("📐 Image size: \(image.size.width) x \(image.size.height)")
        
        // Save original image data first
        guard let originalData = image.jpegData(compressionQuality: 0.8) else {
            print("❌ Could not convert image to JPEG")
            isProcessing = false
            alertMessage = "Could not process image"
            showAlert = true
            return
        }
        
        print("💾 Original image data: \(originalData.count) bytes")
        
        // Send to server for processing
        DigitClassifierService.shared.processImage(image) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let serverDigits):
                    print("✅ Received \(serverDigits.count) digits from server")
                    self.processingStatus = "Processing \(serverDigits.count) digits..."
                    
                    // Convert server results to DetectedDigitData
                    var digitData: [DetectedDigitData] = []
                    var failedDecodes = 0
                    
                    for (index, serverDigit) in serverDigits.enumerated() {
                        // Decode base64 image from server
                        if let imageData = Data(base64Encoded: serverDigit.image) {
                            print("✓ Decoded digit \(index): \(serverDigit.digit) - \(imageData.count) bytes")
                            digitData.append(DetectedDigitData(
                                lineNumber: 0,  // Server returns all on one line
                                digitNumber: index + 1,
                                predictedDigit: serverDigit.digit,
                                confidence: serverDigit.confidence,
                                imageData: imageData
                            ))
                        } else {
                            print("⚠️ Failed to decode base64 for digit \(index)")
                            failedDecodes += 1
                        }
                    }
                    
                    if failedDecodes > 0 {
                        print("⚠️ Failed to decode \(failedDecodes) out of \(serverDigits.count) images")
                    }
                    
                    // Create grayscale version for storage
                    let grayImage = self.convertToGreyscale(image) ?? image
                    var quality: CGFloat = 0.8
                    var grayData = grayImage.jpegData(compressionQuality: quality)
                    
                    while let data = grayData, data.count > 500_000 && quality > 0.1 {
                        quality -= 0.1
                        grayData = grayImage.jpegData(compressionQuality: quality)
                    }
                    
                    guard let finalData = grayData else {
                        print("❌ Could not compress grayscale image")
                        self.isProcessing = false
                        self.alertMessage = "Could not compress image"
                        self.showAlert = true
                        return
                    }
                    
                    print("💾 Final grayscale data: \(finalData.count) bytes")
                    
                    // Save to database
                    let photo = SavedPhoto(
                        originalImageData: originalData,
                        processedImageData: finalData,
                        capturedDate: Date(),
                        fileSizeKB: Double(finalData.count) / 1024.0,
                        detectedDigits: digitData
                    )
                    
                    self.modelContext.insert(photo)
                    
                    // Keep only last 3 photos
                    if self.savedPhotos.count >= 3, let oldest = self.savedPhotos.last {
                        print("🗑️ Deleting oldest photo")
                        self.modelContext.delete(oldest)
                    }
                    
                    do {
                        try self.modelContext.save()
                        print("✅ Photo saved to database")
                        
                        if digitData.isEmpty {
                            self.alertMessage = "Photo saved!\nNo digits detected."
                        } else {
                            let digits = digitData.map { "\($0.predictedDigit)" }.joined()
                            self.alertMessage = "Photo saved!\nDetected \(digitData.count) digits: \(digits)"
                        }
                        self.showAlert = true
                        
                    } catch {
                        print("❌ Database save error: \(error)")
                        self.alertMessage = "Error saving: \(error.localizedDescription)"
                        self.showAlert = true
                    }
                    
                    self.isProcessing = false
                    self.processingStatus = ""
                    
                case .failure(let error):
                    print("❌ Server processing failed: \(error.localizedDescription)")
                    self.isProcessing = false
                    self.processingStatus = ""
                    self.alertMessage = """
                    Server error: \(error.localizedDescription)
                    
                    Make sure:
                    1. Flask server is running on your Mac
                    2. Mac and iPhone on same WiFi
                    3. IP address is correct in code (currently: 192.168.1.172)
                    4. Try accessing http://192.168.1.172:5001 in Safari
                    """
                    self.showAlert = true
                }
            }
        }
    }
    
    func convertToGreyscale(_ image: UIImage) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let width = cgImage.width
        let height = cgImage.height
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let gray = context.makeImage() else { return nil }
        
        return UIImage(cgImage: gray, scale: image.scale, orientation: image.imageOrientation)
    }
}

struct ImagePicker: UIViewControllerRepresentable {
    let sourceType: UIImagePickerController.SourceType
    let onImagePicked: (UIImage) -> Void
    @Environment(\.dismiss) private var dismiss
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = sourceType
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.onImagePicked(image)
            }
            parent.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

#Preview {
    ContentView()
}
