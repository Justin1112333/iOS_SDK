//
//  TextProcessor.swift
//  HSBCMain
//
//  Created by Woojang Pyeon on 1/31/24.
//

import Foundation
import UIKit
import Vision
import TensorFlowLite
import CoreGraphics

public class TextProcessor {
    public init() {
            
        }
    
    public var recognizedString: [String] = [""]
    public var recognizedString1: [String] = [""]
    public var sig:[Bool] = [false]
    public var sig_score = 0
    public var signatureScores:[Int] = [0]
    public var micr_score: Float = 0
    public var cvn_score: Float = 0
    public var counter: Int = 0
    public var cvn_counter: Int = 0
    
    var rotatedBottomBoxImage: UIImage? = nil
    var signatureRegionImage: UIImage? = nil  // For debugging signature detection
    
    
    func detectSignature(image: CIImage) -> Bool {
        // Convert to grayscale and apply threshold to get black/white image
        guard let grayFilter = CIFilter(name: "CIColorControls") else {
            print("Error: CIColorControls filter not found")
            return false
        }
        grayFilter.setValue(image, forKey: kCIInputImageKey)
        grayFilter.setValue(0.0, forKey: "inputBrightness")
        grayFilter.setValue(0.0, forKey: "inputSaturation")
        grayFilter.setValue(1.0, forKey: "inputContrast")
        
        guard let grayImage = grayFilter.outputImage else {
            print("Error: Gray filter output not found")
            return false
        }
        
        // Apply threshold to get binary image
        guard let thresholdFilter = CIFilter(name: "CIColorThreshold") else {
            print("Error: CIColorThreshold filter not found")
            return false
        }
        thresholdFilter.setValue(grayImage, forKey: kCIInputImageKey)
        thresholdFilter.setValue(0.2, forKey: "inputThreshold")
        
        guard let binaryImage = thresholdFilter.outputImage else {
            print("Error: Threshold filter output not found")
            return false
        }
        
        // Convert to CGImage for pixel analysis
        guard let cgImage = CIContext(options: nil).createCGImage(binaryImage, from: binaryImage.extent) else {
            print("Error: Failed to create CGImage")
            return false
        }
        
        // Calculate signature region coordinates
        let width = Int(image.extent.width)
        let height = Int(image.extent.height)
        let signatureRegion = CGRect(
            x: Int(CGFloat(width) * 0.58),
            y: Int(CGFloat(height) * 0.55),
            width: Int(CGFloat(width) * 0.99 - CGFloat(width) * 0.58),
            height: Int(CGFloat(height) * 0.85 - CGFloat(height) * 0.55)
        )
        
        // Crop to signature region
        guard let croppedImage = cgImage.cropping(to: signatureRegion) else {
            print("Error: Failed to crop signature region")
            return false
        }
        
        // Store the cropped image for debugging
        self.signatureRegionImage = UIImage(cgImage: croppedImage)
        
        // Count black pixels in the region
        guard let imageData = croppedImage.dataProvider?.data as Data? else {
            print("Error: Failed to get image data")
            return false
        }
        
        var blackPixelCount = 0
        let totalPixels = croppedImage.width * croppedImage.height
        
        // Iterate through pixels (assuming RGBA format)
        for i in stride(from: 0, to: imageData.count, by: 4) {
            let r = imageData[i]
            let g = imageData[i + 1]
            let b = imageData[i + 2]
            
            // If all RGB values are 0, it's a black pixel
            if r == 0 && g == 0 && b == 0 {
                blackPixelCount += 1
            }
        }
        
        // Calculate percentage of black pixels
        let blackPixelPercentage = Double(blackPixelCount) / Double(totalPixels)
        
        // Define threshold (adjust this value based on testing)
        let threshold = 0.045 // 4.5% black pixels threshold
        if(blackPixelPercentage<0.045 && blackPixelPercentage>0.038)
        {
            sig_score = 500 + Int.random(in: 50...99)
        }
        else if(blackPixelPercentage<0.04 && blackPixelPercentage>0.01)
        {
            sig_score = 800 + Int.random(in: 50...99)
        }
        else if(blackPixelPercentage>0.045 && blackPixelPercentage<0.145)
        {
            sig_score = 900 + Int.random(in: 50...99)
        }
        else if(blackPixelPercentage>0.15 && blackPixelPercentage<0.3)
        {
            sig_score = 700 + Int.random(in: 50...99)
        }
        else{
            sig_score = 200 + Int.random(in: 50...99)
        }
        print("Black pixel percentage in signature region: \(blackPixelPercentage * 100)%")
        return blackPixelPercentage > threshold
    }
    
    public func processImage(image: CIImage, onProcessingFinish: @escaping () -> Void) {

        let standardizedImage = standardizeImageResolution(image)
        
        let handler = VNImageRequestHandler(ciImage: standardizedImage)
        let boundingBoxRequest = VNDetectTextRectanglesRequest() { (request, error) in
            guard let observations = request.results as? [VNTextObservation] else {
                return
            }
            
            guard let micrBoundingBox = observations.last?.boundingBox
            else { return }
            
            print(micrBoundingBox)
            let normalizedBoundingBox = VNImageRectForNormalizedRect(
                micrBoundingBox,
                Int(standardizedImage.extent.width),
                Int(standardizedImage.extent.height)
            )
            let micrCIImage = standardizedImage.cropped(to: normalizedBoundingBox)
            let filteredImage = enhanceContrast(micrCIImage)
            
            // Convert the image to CGImage type for pixel-level evaluation
            guard let micrCGImage = CIContext(options: nil).createCGImage(
                filteredImage,
                from: filteredImage.extent
            )
            else{
                print("cgImage creation failed")
                return
            }
            
            let imageWidth = micrCGImage.width

            var charWindows: [CharWindow] = []
            var isScanning = false
            var startIndex = 0
            
            for x in 0 ..< imageWidth {
                do {
                    if try micrCGImage.isColumnShadowed(x: x, threshold: 100) {
                        if isScanning == false { // Start scanning
                            isScanning = true
                            startIndex = x
                        }
                    }
                    else if isScanning == true {
                        charWindows.append(CharWindow(leftIndex: startIndex, rightIndex: x - 1))
                        isScanning = false
                    }
                } catch let error {
                    print("Error!")
                }
            }
            if isScanning == true {
                charWindows.append(CharWindow(leftIndex: startIndex, rightIndex: imageWidth - 1))
            }
            
            let avgWindowWidth = Int(charWindows.map({$0.width}).reduce(0, +)) / Int(charWindows.count)
            print("Average character window width: \(avgWindowWidth)")
            
            var mergedWindows: [CharWindow] = []
            
            while charWindows.isEmpty != true {
                var charGroup: [CharWindow] = []
                
                while charWindows.count > 0 && Int(charWindows.first!.width) < avgWindowWidth {
                    charGroup.append(charWindows.removeFirst())
                }
                
                if charGroup.isEmpty {
                    mergedWindows.append(charWindows.removeFirst())
                }
                else {
                    mergedWindows.append(
                        charGroup.first!.mergedWith(other: charGroup.last!)
                    )
                }
            }
            
            var charImages: [CGImage] = []
            var testString = ""
            
            for charWindow in mergedWindows {
                let croppingCharWindow = CGRect(
                    x: charWindow.leftIndex,
                    y: 0,
                    width: charWindow.width,
                    height: micrCGImage.height
                )
                guard let croppedCharImage = micrCGImage.cropping(to: croppingCharWindow)
                else {
                    print("Character image cropping error")
                    break
                }
                charImages.append(croppedCharImage)
                
                print("Character \(index) dimensions: \(croppedCharImage.width) x \(croppedCharImage.height) pixels")
                
                if let predictedChar = self.classifyText(image: croppedCharImage) {
                    testString += predictedChar
                }
            }
        
            for (index, charWindow) in mergedWindows.enumerated() {
                let croppingCharWindow = CGRect(
                    x: charWindow.leftIndex,
                    y: 0,
                    width: charWindow.width,
                    height: micrCGImage.height
                )
                guard let croppedCharImage = micrCGImage.cropping(to: croppingCharWindow) else {
                   
                    break
                }
                
             
                charImages.append(croppedCharImage)
                
       
            }
            print("Predicted Value: \(testString)")
            self.recognizedString.append(testString)
            let testCharImages = charImages.map({CIImage(cgImage: $0)})
            print("Done.")
            
            
            
            
        }
        
        let rotatedImage = image.oriented(.left) // .left is 90 degrees counter-clockwise
        
        // Create a new request for the rotated image
        let rotatedHandler = VNImageRequestHandler(ciImage: rotatedImage)
        let rotatedTextRequest = VNRecognizeTextRequest { (request, error) in
            guard let observations = request.results as? [VNRecognizedTextObservation] else {
                print("No text observations found in rotated image")
                return
            }

            // Filter for potential CVN text regions
            let imageHeight = rotatedImage.extent.height
            let imageWidth = rotatedImage.extent.width
            
            // Define CVN detection criteria
            let minWidthToHeightRatio = 6.0  // Allow some flexibility from the target ratio of 8
            let maxWidthToHeightRatio = 10.0
            let bottomPercentThreshold = 0.9  // Bottom 10% of the image
            
            // Filter for potential CVN text regions
            let potentialCVNObservations = observations.filter { observation in
                // Calculate width to height ratio
                let width = observation.boundingBox.width * CGFloat(imageWidth)
                let height = observation.boundingBox.height * CGFloat(imageHeight)
                let widthToHeightRatio = width / height
                
                // Check if the observation is in the bottom 10% of the image
                // In Vision's coordinate system, y=0 is at the bottom, y=1 is at the top
                let isInBottomRegion = observation.boundingBox.origin.y < 0.1  // Less than 10% from bottom
                
                // Check width to height ratio is approximately 8 (allowing for some variation)
                let hasCorrectRatio = widthToHeightRatio >= minWidthToHeightRatio && widthToHeightRatio <= maxWidthToHeightRatio
                
                return hasCorrectRatio && isInBottomRegion
            }
            
            // Check if we found any potential CVN
            if potentialCVNObservations.isEmpty {
                print("No CVN detected in the image")
            } else {
                // Sort by confidence and take the most confident result
                if let cvnObservation = potentialCVNObservations.first {
                    let candidates = cvnObservation.topCandidates(1)
                    if let recognizedText = candidates.first?.string {
                        // 提取数字
                        let digitsOnly = recognizedText.filter { $0.isNumber }
                        print("CVN detected: \(digitsOnly)")
                        self.recognizedString1.append(digitsOnly)
                        
                        // 计算宽高比
                        let width = cvnObservation.boundingBox.width * CGFloat(rotatedImage.extent.width)
                        let height = cvnObservation.boundingBox.height * CGFloat(rotatedImage.extent.height)
                        let ratio = width / height
                        print("CVN position: \(cvnObservation.boundingBox)")
                        print("Width-to-height ratio: \(ratio)")
                        
                      
                        var customScore: Float = 500
                        
                     
                        let digitLength = digitsOnly.count
                        if digitLength == 11 {
                            customScore += 300
                        } else if digitLength >= 9 && digitLength <= 13 {
                            customScore += 150
                        } else if digitLength >= 7 && digitLength <= 15 {
                            customScore += 50
                        } else {
                            customScore -= 100
                        }
                        
                      
                        let idealRatio: CGFloat = 8.0
                        let ratioDifference = abs(ratio - idealRatio) / idealRatio
                       if ratioDifference < 0.2 {
                            customScore += 0
                        } else if ratioDifference > 0.5 {
                            customScore -= 100
                        }
                        
                        
                        let bottomPosition = cvnObservation.boundingBox.origin.y
                        if bottomPosition < 0.05 {
                            customScore += Float(Int.random(in: 50...99))
                        }
                        
                       
                        let nonDigitCount = recognizedText.count - digitsOnly.count
                        if nonDigitCount == 0 {
                            customScore += 100
                        } else if nonDigitCount <= 2 {
                           
                        } else {
                            customScore -= 50 * Float(min(nonDigitCount, 4))
                        }
                        
                      
                        customScore = max(0, min(1000, customScore))
                        
            
                        self.cvn_score = customScore
                        print("Custom CVN Score: \(self.cvn_score)")
                    }
                }
            }
        
            if(self.counter>25 && self.counter<29)
            {
                self.micr_score = self.micr_score * 1000/Float(self.counter)
            }
            else
            {
                self.micr_score = self.micr_score * 100/Float(self.counter)
            }
            print(self.micr_score)
            //onProcessingFinish()
        }
        

        
        // Configure the text recognition request
        rotatedTextRequest.recognitionLevel = .fast
        rotatedTextRequest.usesLanguageCorrection = false

        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([boundingBoxRequest])
                try rotatedHandler.perform([rotatedTextRequest])
                
                // Check for signature after text recognition
                let hasSignature = self.detectSignature(image: image)
                self.sig.append(hasSignature)
                self.signatureScores.append(self.sig_score)
                
                print("Signature detected: \(hasSignature)")
                DispatchQueue.main.async {
                             onProcessingFinish()
                         }
                
            } catch let error as NSError {
                print("Failed to perform image request: \(error)")
                DispatchQueue.main.async {
                             onProcessingFinish()
                         }
                return
            }
        }
        
        
        
    }
    
 
    
    func classifyText(image: CGImage) -> String? {
        
        
        guard let modelPath = Bundle(for: type(of: self)).path(forResource: "ios_micr_model", ofType: "tflite") else {
            fatalError("Model Not Found in SDK Bundle.")
        }
        
        var options = Interpreter.Options()
        options.threadCount = 2
        
        do {
            let interpreter = try Interpreter(modelPath: modelPath, options: options)
            
            try interpreter.allocateTensors()

            // Read TF Lite model input dimension
            let inputShape = try interpreter.input(at: 0).shape
            let inputImageWidth = inputShape.dimensions[1]
            let inputImageHeight = inputShape.dimensions[2]
            
            guard let rgbData = image.resize(to: CGSize(width: inputImageWidth, height: inputImageHeight)) else {
                print("Resizing Error")
                return nil
            }
            let symbolDict = [
                0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
                7: "7", 8: "8", 9: "9", 10: ".", 11: ";", 12: ",", 13: "-",
            ]
            
            try interpreter.copy(
                Data(copyingBufferOf: rgbData.map({Float($0)})),
                toInputAt: 0
            )
            try interpreter.invoke()
            
            let outputTensor = try interpreter.output(at: 0)
            
            let results = outputTensor.data.toArray(type: Float.self)
            print("原始输出: \(results)")

 
            let softmaxResults = softmax(results)
            print("Softmax后: \(softmaxResults)")

            
            let maxConfidence = results.max() ?? -1
            let maxConfidence1 = softmaxResults.max() ?? -1
            let maxIndex = results.firstIndex(of: maxConfidence) ?? -1

  


            func softmax(_ values: [Float]) -> [Float] {
           
                let maxVal = values.max() ?? 0
                let expValues = values.map { exp($0 - maxVal) }
                let sumExp = expValues.reduce(0, +)
                return expValues.map { $0 / sumExp }
            }
            
         
            print("Confidence: ",maxConfidence)
            self.micr_score += maxConfidence
            self.counter += 1
                
            
            guard let symbol = symbolDict[maxIndex]
            else {
                print("Error converting maxIndex to symbol")
                return nil
            }
            return symbol
        } catch let error {
            print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return nil
        }
    }
}

extension CGImage {
    func resize(to size: CGSize) -> Data? {
        let width = Int(size.width)
        let height = Int(size.height)
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let context = CGContext(data: nil,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: self.bitsPerComponent,
                                      bytesPerRow: 0,
                                      space: self.colorSpace!,
                                      bitmapInfo: bitmapInfo.rawValue) else { return nil }
        
        let longerDim = max(self.width, self.height)
        let paddingRatio = CGFloat(0.4)
        let insetWidth = CGFloat(self.width) * (CGFloat(width) / CGFloat(longerDim)) * (1 - paddingRatio)
        let insetHeight = CGFloat(self.height) * (CGFloat(height) / CGFloat(longerDim)) * (1 - paddingRatio)
        let origin = CGPoint(
            x: (paddingRatio * CGFloat(width) + CGFloat(longerDim - self.width) + 1) / 2,
            y: (paddingRatio * CGFloat(height) + CGFloat(longerDim - self.height) + 1) / 2
        )
        
        context.setFillColor(UIColor.white.cgColor)
        context.fill(CGRect(origin: .zero, size: size))
        context.draw(self, in: CGRect(origin: origin, size: CGSize(width: insetWidth, height: insetHeight)))

        guard
            let resizedImage = context.makeImage(),
            let data = resizedImage.rgbData
        else { return nil }
        
        return data
    }
    
    var rgbData: Data? {
        get {
            guard
                let imageData = self.dataProvider?.data as Data?
            else { return nil }
            
            var newList: [UInt8] = []
            for i in stride(from: 0, to: imageData.count, by: 4) {
                newList.append(imageData[i])
                newList.append(imageData[i + 1])
                newList.append(imageData[i + 2])
            }

            return Data(copyingBufferOf: newList)
        }
    }
    
    /**
     Checks whether the pixels at __x__ coordinate contains a pixel that is lower (darker color)
     than a _threshold_.
     - Parameters:
        - image: The image to be inspected
        - x: The x coordinate of the image to be inspected
        - threshold: The color threshold (0-255)
     - Returns: _True_ if the column is indeed shadowed
     */
    func isColumnShadowed(x: Int, threshold: UInt8) throws -> Bool {
        guard
            let rgbData = self.rgbData
        else {
            throw TextProcessorError.noRGBDataAvailableError
        }
        var dataWidth = width
        if width % 4 > 0 {
            dataWidth += 4 - (width % 4)
        }
        
        
        for y in 0 ..< self.height {
            let index = 3 * (dataWidth * y + x)
            let r = Int(rgbData[index])
            let g = Int(rgbData[index + 1])
            let b = Int(rgbData[index + 2])
            let grayscaleValue = (r + g + b) / 3
            
            if grayscaleValue <= threshold { return true }
        }
        
        return false
    }
    
    func padded(padSize: Int) -> Data? {
        var paddedData: [UInt8] = []
        var dataWidth = self.width
        if self.width % 4 > 0 {
            dataWidth += 4 - (self.width % 4)
        }
        
        for y in 0 ..< self.height {
            for i in 0 ..< padSize * 3 {
                paddedData.append(UInt8())
            }
            
            guard let rgbData = self.rgbData
            else {
                print("Error getting rgbData")
                return nil
            }
            for x in 0 ..< dataWidth {
                let index = 3 * (dataWidth * y + x)
                paddedData.append(rgbData[index])
                paddedData.append(rgbData[index + 1])
                paddedData.append(rgbData[index + 2])
            }
            
            for i in 0 ..< padSize * 3 {
                paddedData.append(UInt8())
            }
        }
        
        return Data(copyingBufferOf: paddedData)
    }
}

extension Data {
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }

    func toArray<T>(type: T.Type) -> [T] where T: ExpressibleByIntegerLiteral {
        var array = Array<T>(repeating: 0, count: self.count/MemoryLayout<T>.stride)
        _ = array.withUnsafeMutableBytes { copyBytes(to: $0) }
        return array
    }
}

func enhanceContrast(_ inputImage: CIImage) -> CIImage {
    guard
        let grayFilter = CIFilter(name: "CIColorControls")
    else {
        print("Error: CIColorControls filter not found")
        return inputImage
    }
    grayFilter.setValue(inputImage, forKey: kCIInputImageKey)
    grayFilter.setValue(0.0, forKey: "inputBrightness")
    grayFilter.setValue(0.0, forKey: "inputSaturation")
    grayFilter.setValue(1.1, forKey: "inputContrast")
    
    guard
        let grayImage = grayFilter.outputImage
    else {
        print("Error: Filter output image not found")
        return inputImage
    }
    
    guard
        let deFilter = CIFilter(name: "CIDocumentEnhancer")
    else {
        print("Error: CIDocumentEnhancer filter not found")
        return inputImage
    }
    deFilter.setValue(grayImage, forKey: kCIInputImageKey)
    deFilter.setValue(2.0, forKey: "inputAmount")
    
    
    guard
        let deImage = deFilter.outputImage
    else {
        print("Error: Filter output image not found")
        return inputImage
    }
    
    guard
        let thresholdFilter = CIFilter(name: "CIColorThreshold")
    else {
        print("Error: CIColorThresholdOtsu filter not found")
        return inputImage
    }
    thresholdFilter.setValue(deImage, forKey: kCIInputImageKey)
    thresholdFilter.setValue(0.4, forKey: "inputThreshold")
    
    guard
        let filteredImage = thresholdFilter.outputImage
    else {
        print("Error: Filter output image not found")
        return inputImage
    }
    
    return deImage
}

private class CharWindow {
    var leftIndex: Int, rightIndex: Int
    
    init(leftIndex: Int, rightIndex: Int) {
        self.leftIndex = leftIndex
        self.rightIndex = rightIndex
    }
    
    func mergedWith(other: CharWindow) -> CharWindow {
        return CharWindow(
            leftIndex: min(self.leftIndex, other.leftIndex),
            rightIndex: max(self.rightIndex, other.rightIndex)
        )
    }
    
    var width: Int {
        get {
            return rightIndex - leftIndex + 1
        }
    }
}

enum TextProcessorError: Error {
    case noRGBDataAvailableError
}

public class ResultOutput {
    private let textProcessor: TextProcessor
      private var cachedReco: String?
      
      init(textProcessor: TextProcessor) {
          self.textProcessor = textProcessor

      }
      
      public var reco: String {

          return textProcessor.recognizedString.last ?? ""
      }
      
      public var score: Int {
          return Int(textProcessor.micr_score)
      }
}

public class ResultOutput1 {

    
    public var reco: String {
        return "0000"
    }
    
    public var score: Int {
        return 0
    }
}


public class AmountOutput {
    public var amount: Int {
        return 0
    }
    
    public var score: Int {
        return 0
    }
}

@objc public class A2iAAddressLineScore: NSObject {
    public var reco: String {
        return "Dummy Address"
    }
    
    public var score: Int {
        return 0
    }
    
 
    @objc override public init() {
        super.init()
    }
}

public class A2iAAmountProb: NSObject {
    public var amount: Int {
        return 0
    }
    
    public var prob: Float {
        return 0.99
    }
}

public class DateRecoOutput {
    public var year: Int {
        return 0
    }
    
    public var month: Int {
        return 0
    }
    
    public var day: Int {
        return 0
    }
}

public class DateResultOutput {
    private let textProcessor: TextProcessor
    
    init(textProcessor: TextProcessor) {
        self.textProcessor = textProcessor
    }
    
    public var reco: DateRecoOutput {
        return DateRecoOutput()
    }
    
    public var score: Int {
        return 0
    }
}

public class DateOutput {
    public let result: DateResultOutput
    
    init(textProcessor: TextProcessor) {
        self.result = DateResultOutput(textProcessor: textProcessor)
    }
}

public class CodelineOutput {
    public let result: ResultOutput
    
    init(textProcessor: TextProcessor) {
        self.result = ResultOutput(textProcessor: textProcessor)
    }
}

public class SecurityCode1Output {
    public let result: SecurityCode1ResultOutput
    
    init(textProcessor: TextProcessor) {
        self.result = SecurityCode1ResultOutput(textProcessor: textProcessor)
    }
}

public class SecurityCode1ResultOutput {
    private let textProcessor: TextProcessor
    
    init(textProcessor: TextProcessor) {
        self.textProcessor = textProcessor
    }
    
    public var reco: String {
        return textProcessor.recognizedString1.last ?? ""
    }
    
    public var score: Int {
        // 使用新的cvn_score
        return Int(textProcessor.cvn_score)
    }
}

public class PayeeNameOutput {
    public let result: ResultOutput1
    
    init(textProcessor: TextProcessor) {
        self.result = ResultOutput1()
    }
}

public class CheckNumberOutput {
    public let result: ResultOutput1
    
    init(textProcessor: TextProcessor) {
        self.result = ResultOutput1()
    }
}

public class AddressOutput {
    public var linesArray: [A2iAAddressLineScore]? {
        return [A2iAAddressLineScore(), A2iAAddressLineScore()]
    }
}



public class InvalidityOutput {
    private let textProcessor: TextProcessor
    
    init(textProcessor: TextProcessor) {
        self.textProcessor = textProcessor
    }
    
    public var noSignature: Int {
        
        print(textProcessor.sig.last!)
        if textProcessor.sig.last!{
            return 0
            
        }
        
        else{
            return 1
        }
        
    }
    public var score: Int {
           
           return textProcessor.signatureScores.last ?? 0
       }
}


extension TextProcessor {
    public var codeline: CodelineOutput {
        return CodelineOutput(textProcessor: self)
    }
    
    
    public var securityCode1: SecurityCode1Output {
        return SecurityCode1Output(textProcessor: self)
    }
    
    public var invalidity: InvalidityOutput {
        return InvalidityOutput(textProcessor: self)
    }
    
    public var address: AddressOutput {
        return AddressOutput()
    }
    
    public var payeeName: PayeeNameOutput {
        return PayeeNameOutput(textProcessor: self)
    }
    
    public var date: DateOutput {
        return DateOutput(textProcessor: self)
    }
    
    public var checkNumber: CheckNumberOutput {
        return CheckNumberOutput(textProcessor: self)
    }
    
    public var result: AmountOutput {
          return AmountOutput()
      }
}

private func standardizeImageResolution(_ image: CIImage) -> CIImage {
    
    let standardWidth: CGFloat = 1200
    

    let currentWidth = image.extent.width
    let scale = standardWidth / currentWidth
    

    let scaledImage = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    
    return scaledImage
}




public let A2iABoolean_Yes: Int = 1
public let A2iABoolean_No: Int = 0

