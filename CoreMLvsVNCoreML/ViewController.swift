//
//  ViewController.swift
//  CoreMLvsVNCoreML
//
//  Created by Kang Paul on 2018/10/6.
//  Copyright © 2018年 Kang Paul. All rights reserved.
//

import UIKit
import AVKit
import Vision
//import CoreML *不用 import 也可以

let CoreML = true //選擇使用 CoreML 或 VNCoreML 方式

class ViewController: UIViewController,
AVCaptureVideoDataOutputSampleBufferDelegate {
    
    let model = Inceptionv3()
    
    let dataOutput = AVCaptureVideoDataOutput()
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        //MARK: - 1. Setup the camera, craete a captureSession
        let captureSession = AVCaptureSession()
        
        captureSession.sessionPreset = .photo
        guard let captureDevice = AVCaptureDevice.default(for: .video)
            else {return}
        
        guard let input = try? AVCaptureDeviceInput(device:
            captureDevice) else {return}
        
        captureSession.addInput(input)
        captureSession.startRunning()
        
        //MARK: - 2. Setup the review layer
        let previewLayer = AVCaptureVideoPreviewLayer(session:
            captureSession)
        view.layer.addSublayer(previewLayer)
        previewLayer.frame = view.frame
        //到這裡就可以在 app 中顯示 camera 影像
        
        //MARK: -3. 設定 dataOutput delegate 及其他設定
        dataOutput.setSampleBufferDelegate(self,
                                           queue: DispatchQueue(label: "videoQueue"))
        
        if CoreML {
            //使用 VNCoreML 不需要，但使用 CoreML 需要設定 formatType，
            //不然會有以下 error: 讓我一開始卡了很久
            //Error Domain=com.apple.CoreML Code=1 "Image is not expected
            //type 32BGRA or 32ARGB, instead is Unsupported (875704422)"
            dataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey
                as AnyHashable as! String:
                NSNumber(value: kCVPixelFormatType_32BGRA)]
        }
        
        captureSession.addOutput(dataOutput)
    }
    
    
    //對影像進行處理，"應該是"每個 frame 都會呼叫 captureOutput delegate
    //method，影像用 sampleBuffer 傳入處理
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        
        //設定用 CoreML 或 VNCoreML 處理
        if CoreML {
            captureOutputCoreML(output, didOutput:sampleBuffer,
                                from:connection)
        } else {
            captureOutputVNCoreML(output, didOutput:sampleBuffer,
                                  from:connection)
        }
    }
    
    // CoreML: delegate didOutput
    func captureOutputCoreML(_ output: AVCaptureOutput,
                             didOutput sampleBuffer: CMSampleBuffer,
                             from connection: AVCaptureConnection) {
        
        //convert sampleBuffer type from CMSampleBuffer to cvPixelBuffer
        guard let pixelBuffer =
            CMSampleBufferGetImageBuffer(sampleBuffer)
            else { return }
        
        do {
            //resize image to the domension defined in InceptionV3model
            let resizedImage = self.resize(pixelBuffer: pixelBuffer)!
            
            let prediction = try model.prediction(image: resizedImage)
            
            //[Optional] Use DispatchQueue.main.async to reduce the
            //response lag
            DispatchQueue.main.async {
                if let prob =
                    prediction.classLabelProbs[prediction.classLabel] {
                    
                    //[Optional] show results on UILabel
                    print("CoreML:\(prediction.classLabel) \(String(describing: prob))")
                }
            }
        }
        catch let error as NSError {
            fatalError("Unexpected error ocurred: \(error.localizedDescription).")
        }
    }
    
    // VNCoreML: delegate didOutput
    func captureOutputVNCoreML(_ output: AVCaptureOutput,
                               didOutput sampleBuffer: CMSampleBuffer,
                               from connection: AVCaptureConnection) {
        
        //convert sampleBuffer type from CMSampleBuffer to cvPixelBuffer
        guard let pixelBuffer =
            CMSampleBufferGetImageBuffer(sampleBuffer)
            else { return }
        //MARK: - Using VNCoreMLModel
        guard let model = try? VNCoreMLModel(for: Inceptionv3().model)
            else {return}
        
        let requests = VNCoreMLRequest(model: model) {
            (finishedReq, err) in
            guard let results = finishedReq.results as?
                [VNClassificationObservation] else {return}
            
            guard let firstObservation = results.first else {return}
            
            print("VNCoreML ", firstObservation.identifier,
                  firstObservation.confidence)
        }
        
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                   options: [:]).perform([requests])
    }
    
    /// https://github.com/yulingtianxia/Core-ML-Sample)
    /// Only for CoreML
    /// resize CVPixelBuffer (from
    /// - Parameter pixelBuffer: CVPixelBuffer by camera output
    /// - Returns: CVPixelBuffer with size (299, 299)
    func resize(pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        let imageSide = 299
        var ciImage = CIImage(cvPixelBuffer: pixelBuffer, options: nil)
        
        let transform = CGAffineTransform(
            scaleX: CGFloat(imageSide)/CGFloat(CVPixelBufferGetWidth(pixelBuffer)),
            y: CGFloat(imageSide)/CGFloat(CVPixelBufferGetHeight(pixelBuffer)))
        
        ciImage = ciImage.transformed(by: transform).cropped(
            to: CGRect(x: 0, y: 0, width: imageSide, height: imageSide))
        
        let ciContext = CIContext()
        
        var resizeBuffer: CVPixelBuffer?
        
        CVPixelBufferCreate(kCFAllocatorDefault, imageSide, imageSide,
                            CVPixelBufferGetPixelFormatType(pixelBuffer),
                            nil, &resizeBuffer)
        
        ciContext.render(ciImage, to: resizeBuffer!)
        
        return resizeBuffer
    }
}


