let ws = null;
let webcam = null;

function renderCharts(sig) {
    // Very simple chart render on canvas for skeleton demonstration phase
    const cSig = document.getElementById('signal-chart');
    const ctxSig = cSig.getContext('2d');
    
    ctxSig.clearRect(0,0, cSig.width, cSig.height);
    
    const filt = sig.filtered;
    if (filt && filt.length > 0) {
        ctxSig.beginPath();
        ctxSig.strokeStyle = '#00d2ff';
        const minV = Math.min(...filt);
        const maxV = Math.max(...filt);
        const range = maxV - minV || 1;
        
        for (let i = 0; i < filt.length; i++) {
            const x = (i / filt.length) * cSig.width;
            const y = cSig.height - ((filt[i] - minV) / range) * cSig.height;
            if (i === 0) ctxSig.moveTo(x, y);
            else ctxSig.lineTo(x, y);
        }
        ctxSig.stroke();
    }
}

function onOpenCvReady() {
    console.log("OpenCV initialized via script load.");
    webcam = new WebcamManager();
    webcam.initOpenCV();

    const host = window.location.host;
    ws = new VPWebSocket(`ws://${host}/ws/vitals`, 
        (data) => {
            // on result
            document.getElementById('hr-val').innerText = data.hr_bpm ? data.hr_bpm.toFixed(1) : '--';
            document.getElementById('conf-val').innerText = (data.confidence * 100).toFixed(0);
            document.getElementById('progress-bar').style.width = '100%';
            document.getElementById('message-val').innerText = `Tracking (Model: ${data.model})`;
            if(data.signals) renderCharts(data.signals);
        },
        (data) => {
            // on warmup
            document.getElementById('progress-bar').style.width = `${data.progress * 100}%`;
            document.getElementById('message-val').innerText = `Warming up Buffer...`;
        },
        (data) => {
            // on warning
            document.getElementById('message-val').innerText = data.message;
        }
    );

    document.getElementById('start-btn').addEventListener('click', () => {
        webcam.startCamera();
        webcam.setOnFrame((b64, ts) => ws.sendFrame(b64, ts));
        document.getElementById('start-btn').disabled = true;
    });
}
