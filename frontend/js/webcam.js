class WebcamManager {
    constructor() {
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('output-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.cropCanvas = document.getElementById('crop-canvas');
        this.cropCtx = this.cropCanvas.getContext('2d', { willReadFrequently: true });
        
        this.faceCascade = null;
        this.srcMat = null;
        this.grayMat = null;
        this.faces = null;
        
        this.isRunning = false;
        this.onFrameCallback = null;
        
        this.fpsCount = 0;
        this.lastFpsTime = Date.now();
        
        this.lastFaceBox = null;
    }

    async initOpenCV() {
        try {
            this.faceCascade = new cv.CascadeClassifier();
            this.faceCascade.load('haarcascade_frontalface_default.xml');
            console.log("OpenCV initialized");
        } catch(e) {
            console.error("Failed loading HaarCascade", e);
        }
    }

    async startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, frameRate: { ideal: 30 } }
            });
            this.video.srcObject = stream;
            
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    this.canvas.width = this.video.videoWidth;
                    this.canvas.height = this.video.videoHeight;
                    
                    this.srcMat = new cv.Mat(this.video.videoHeight, this.video.videoWidth, cv.CV_8UC4);
                    this.grayMat = new cv.Mat(this.video.videoHeight, this.video.videoWidth, cv.CV_8UC1);
                    this.faces = new cv.RectVector();
                    
                    resolve();
                };
            });
            
            this.isRunning = true;
            this.processFrame();
        } catch(e) {
            console.error("Camera error", e);
            alert("Could not access camera.");
        }
    }

    processFrame() {
        if (!this.isRunning) return;

        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        let imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        this.srcMat.data.set(imageData.data);

        cv.cvtColor(this.srcMat, this.grayMat, cv.COLOR_RGBA2GRAY);
        
        // Detect face
        let maxSize = new cv.Size(0,0);
        let minSize = new cv.Size(100,100);
        this.faceCascade.detectMultiScale(this.grayMat, this.faces, 1.1, 3, 0, minSize, maxSize);

        let faceRect = null;
        if (this.faces.size() > 0) {
            // Find largest face
            let maxArea = 0;
            for (let i = 0; i < this.faces.size(); ++i) {
                let face = this.faces.get(i);
                if (face.width * face.height > maxArea) {
                    maxArea = face.width * face.height;
                    faceRect = face;
                }
            }
        }

        // Smoothing face box
        if (faceRect) {
            if (!this.lastFaceBox) this.lastFaceBox = {x: faceRect.x, y: faceRect.y, w: faceRect.width, h: faceRect.height};
            else {
                this.lastFaceBox.x = this.lastFaceBox.x * 0.8 + faceRect.x * 0.2;
                this.lastFaceBox.y = this.lastFaceBox.y * 0.8 + faceRect.y * 0.2;
                this.lastFaceBox.w = this.lastFaceBox.w * 0.8 + faceRect.width * 0.2;
                this.lastFaceBox.h = this.lastFaceBox.h * 0.8 + faceRect.height * 0.2;
            }
        }

        if (this.lastFaceBox) {
            const fb = this.lastFaceBox;
            this.ctx.strokeStyle = '#00d2ff';
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(fb.x, fb.y, fb.w, fb.h);

            // Crop face
            let cropW = Math.max(1, fb.w);
            let cropH = Math.max(1, fb.h);
            this.cropCtx.drawImage(
                this.video,
                Math.max(0, fb.x), Math.max(0, fb.y), cropW, cropH,
                0, 0, 72, 72
            );

            // Trigger callback
            if (this.onFrameCallback) {
                const dataUrl = this.cropCanvas.toDataURL('image/jpeg', 0.8);
                this.onFrameCallback(dataUrl, Date.now() / 1000.0);
            }
        }

        // FPS
        this.fpsCount++;
        let now = Date.now();
        if (now - this.lastFpsTime >= 1000) {
            document.getElementById('fps-val').innerText = this.fpsCount;
            this.fpsCount = 0;
            this.lastFpsTime = now;
        }

        requestAnimationFrame(() => this.processFrame());
    }

    setOnFrame(callback) {
        this.onFrameCallback = callback;
    }
}
