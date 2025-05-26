import Foundation
import Vision
import CoreML
import UIKit
import CoreGraphics
import Accelerate

public class Verify {

    private var coreMLModel: MLModel?
    private let targetSize = CGSize(width: 64, height: 64)
    
    public init(modelName: String) {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all

            let sdkBundle = Bundle(for: type(of: self))
            if let modelURL = sdkBundle.url(forResource: modelName, withExtension: "mlmodelc") {
                coreMLModel = try MLModel(contentsOf: modelURL, configuration: config)
                print("Model loaded successfully from SDK Bundle")
            } else {
                print("Model file \(modelName) not found in SDK Bundle")
            }
        } catch {
            print("Error loading model: \(error)")
        }
    }
    
    func sigmoid(_ x: Float) -> Float {
        return 1 / (1 + exp(-x))
    }
    
    public func predict(image: CGImage) -> Int? {
        guard let processedImage = preprocessImage(image) else {
            print("Image preprocessing failed")
            return nil
        }
        
        guard let inputArray = processedImage.toNormalizedAndTransposedMLMultiArray() else {
            print("Failed to convert image to MLMultiArray")
            return nil
        }
        
        do {
            let input = ChequeInput(input: inputArray)
            guard let output = try coreMLModel?.prediction(from: input),
                  let result = output.featureValue(for: "output")?.int64Value else {
                print("Failed to get model prediction")
                return nil
            }
            
            let intResult = Int(result)
            print("Prediction result: \(intResult)")
            return intResult
        } catch {
            print("Prediction error: \(error)")
            return nil
        }
    }
    private func preprocessImage(_ image: CGImage) -> CGImage? {
        guard let grayscaleImage = convertToGrayscale(image: image) else {
            print("Failed to convert to Grayscale")
            return nil
        }
        
        guard let resizedImage = resizeImage(image: grayscaleImage, targetSize: targetSize) else {
            print("Failed to resize image")
            return nil
        }
        
        return resizedImage
    }
    
    /// Convert image to Grayscale
    private func convertToGrayscale(image: CGImage) -> CGImage? {
        let context = CGContext(
            data: nil,
            width: image.width,
            height: image.height,
            bitsPerComponent: 8,
            bytesPerRow: image.width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )
        
        context?.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        
        guard let pic = context?.makeImage() else
        {
            print("NA")
            return nil
        }
        
        print(pic.bitsPerPixel)
        return context?.makeImage()
    }
    
    /// Resize image to target size
    func resizeImage(image: CGImage, targetSize: CGSize) -> CGImage? {
        let context = CGContext(
            data: nil,
            width: Int(targetSize.width),
            height: Int(targetSize.height),
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )
        
        context?.interpolationQuality = .medium
        context?.draw(image, in: CGRect(origin: .zero, size: targetSize))
        
        return context?.makeImage()
    }
}

class ChequeInput: MLFeatureProvider {
    var input: MLMultiArray
    var featureNames: Set<String> { ["input"] }
    
    init(input: MLMultiArray) {
        self.input = input
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard featureName == "input" else { return nil }
        return MLFeatureValue(multiArray: input)
    }
}

extension CGImage {
    /// Convert grayscale image to flattened MLMultiArray (4096) using double precision
    func toNormalizedAndTransposedMLMultiArray() -> MLMultiArray? {
        let width = self.width
        let height = self.height
        guard width == 64 && height == 64 else {
            print("Image dimensions are not 64x64")
            return nil
        }
        
        guard let dataProvider = self.dataProvider,
              let data = dataProvider.data,
              let bytes = CFDataGetBytePtr(data) else {
            print("Failed to get image data")
            return nil
        }
        

        let shape = [4096] as [NSNumber]
        guard let array = try? MLMultiArray(shape: shape, dataType: .double) else {
            print("Failed to create MLMultiArray")
            return nil
        }
        
        for y in 0..<64 {
            for x in 0..<64 {
                let pixelOffset = y * width + x
                let flatIndex = y * 64 + x
                
                let pixelValue = Double(bytes[pixelOffset])
                
                array[flatIndex] = NSNumber(value: pixelValue)
            }
        }
        return array
    }
}
