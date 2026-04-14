class VPWebSocket {
    constructor(url, onResult, onWarmup, onWarning) {
        this.url = url;
        this.onResult = onResult;
        this.onWarmup = onWarmup;
        this.onWarning = onWarning;
        this.isConnecting = true;
        
        this.connect();
    }

    connect() {
        this.ws = new WebSocket(this.url);
        this.ws.onopen = () => {
            console.log('Connected to VitalSense Server');
            document.getElementById('status').className = 'badge connected';
            document.getElementById('status').innerText = 'Connected';
            document.getElementById('start-btn').disabled = false;
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'result') {
                    this.onResult(data);
                } else if (data.type === 'warmup') {
                    this.onWarmup(data);
                } else if (data.type === 'quality_warning') {
                    this.onWarning(data);
                }
            } catch(e) {
                console.error("Message parse error", e);
            }
        };

        this.ws.onerror = (e) => {
            console.error('WebSocket Error', e);
        };

        this.ws.onclose = () => {
            console.log('Disconnected reconnecting in 3s...');
            document.getElementById('status').className = 'badge error';
            document.getElementById('status').innerText = 'Disconnected';
            document.getElementById('start-btn').disabled = true;
            setTimeout(() => this.connect(), 3000);
        };
    }

    sendFrame(frameDataUrl, timestamp) {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'frame',
                data: frameDataUrl,
                timestamp: timestamp
            }));
        }
    }
}
