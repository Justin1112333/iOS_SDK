//
//  ImageProcessor.swift
//  AIFinDemoApp
//
//  Created by Woojang Pyeon on 1/15/24.
//

import Foundation
import UIKit
import Vision

public class ImageProcessor {
    
    public init() {
            
        }
    
    public var processedImage: UIImage? = nil
    public var rectangleObservation: VNRectangleObservation? = nil
    
    public func processImage(image: CIImage, onProcessingFinish: @escaping () -> Void) {
        let handler = VNImageRequestHandler(ciImage: image, options: [:])
        let request = VNDetectDocumentSegmentationRequest(
            completionHandler: { (request, error) in
                guard let observations = request.results as? [VNRectangleObservation] else {
                    return
                }
                self.rectangleObservation = observations[0]
                onProcessingFinish()
                print("Processing complete SDK Tensor Test")
            }
        )
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch let error as NSError {
                print("Failed to perform image request: \(error)")
                return
            }
        }
    }
    
    public func getSkewedImage(image: CIImage) -> CIImage? {
        guard
            let observation = rectangleObservation
        else {
            return nil
        }

        let imageSize = image.extent.size
        
        let boundingBox = observation.boundingBox.scaled(to: imageSize)
        let topLeft = observation.topLeft.scaled(to: imageSize)
        let topRight = observation.topRight.scaled(to: imageSize)
        let bottomLeft = observation.bottomLeft.scaled(to: imageSize)
        let bottomRight = observation.bottomRight.scaled(to: imageSize)
        
        print("Corner values:")
        print(topLeft)
        print(topRight)
        print(bottomLeft)
        print(bottomRight)
        print("Bounding box:")
        print(boundingBox)
        print("Confidence: \(observation.confidence)")
        
        //  TODO: only rectify image when confidence is higher than a specific threshold
        
        let rectifiedImage = image
            .cropped(to: boundingBox)
            .applyingFilter("CIPerspectiveCorrection", parameters: [
                "inputTopLeft": CIVector(cgPoint: topLeft),
                "inputTopRight": CIVector(cgPoint: topRight),
                "inputBottomLeft": CIVector(cgPoint: bottomLeft),
                "inputBottomRight": CIVector(cgPoint: bottomRight)
            ])
        

        let width = rectifiedImage.extent.width
        let height = rectifiedImage.extent.height
        
      
        if height > width {
            return rectifiedImage.oriented(.right)
        }

        return rectifiedImage
        
//        let context = CIContext()
//        let cgImage = context.createCGImage(rectifiedImage, from: rectifiedImage.extent)
//
//        let uiImage = UIImage(cgImage: cgImage!)
//        return uiImage
    }
}

//  Helper extensions
extension CGPoint {
    func scaled(to: CGSize) -> CGPoint {
        return CGPoint(x: self.x * to.width, y: self.y * to.height)
    }
}

extension CGRect {
    func scaled(to: CGSize) -> CGRect {
        return CGRect(
            x: self.origin.x * to.width,
            y: self.origin.y * to.height,
            width: self.width * to.width,
            height: self.height * to.height
        )
    }
}
