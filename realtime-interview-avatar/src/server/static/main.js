// ============================================================================
// Realtime Interview Avatar - Frontend
// WebSocket 기반 실시간 아바타 렌더링 및 오디오 처리
// ============================================================================

// ============================================================================
// State Management
// ============================================================================

const AppState = {
    // Connection state
    sessionId: null,
    wsConnection: null,
    isConnected: false,

    // Interview state
    isInterviewActive: false,
    isMicEnabled: false,
    interviewStage: 'idle',
    questionCount: 0,
    startTime: null,

    // Audio state
    audioContext: null,
    mediaStream: null,
    audioProcessor: null,
    analyser: null,

    // Avatar state
    avatarCanvas: null,
    avatarCtx: null,
    lastFrameTime: 0,
    frameQueue: [],

    // UI state
    theme: 'light',

    // Timers
    elapsedTimeInterval: null,
    volumeCheckInterval: null,

    // Audio playback queue
    audioQueue: [],
    isPlayingAudio: false,
};

// ============================================================================
// Configuration
// ============================================================================

const Config = {
    // Audio settings
    sampleRate: 16000,
    bufferSize: 4096,
    channels: 1,

    // WebSocket settings
    reconnectAttempts: 3,
    reconnectDelay: 2000,

    // Avatar settings
    targetFps: 25,
    frameWidth: 512,
    frameHeight: 512,
};

// ============================================================================
// API Layer
// ============================================================================

const API = {
    baseUrl: window.location.origin,

    async startInterview() {
        const response = await fetch(`${this.baseUrl}/api/interview/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: 'user_' + Date.now(),
                avatar_image: 'default',
                config: {
                    language: 'ko',
                    enable_face_enhancement: false,
                }
            })
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || `Failed to start interview: ${response.statusText}`);
        }

        return await response.json();
    },

    async endInterview(sessionId) {
        const response = await fetch(`${this.baseUrl}/api/interview/${sessionId}/end`, {
            method: 'POST',
        });

        if (!response.ok) {
            throw new Error(`Failed to end interview: ${response.statusText}`);
        }

        return await response.json();
    },

    async getStatus(sessionId) {
        const response = await fetch(`${this.baseUrl}/api/interview/${sessionId}/status`);

        if (!response.ok) {
            throw new Error(`Failed to get status: ${response.statusText}`);
        }

        return await response.json();
    },
};

// ============================================================================
// WebSocket Manager
// ============================================================================

const WebSocketManager = {
    ws: null,
    reconnectAttempts: 0,

    connect(sessionId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // 실시간 오디오 WebSocket 엔드포인트 사용
        const wsUrl = `${protocol}//${window.location.host}/ws/realtime/${sessionId}`;

        console.log('Connecting to WebSocket:', wsUrl);

        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            AppState.isConnected = true;
            UI.updateConnectionStatus('connected');
        };

        this.ws.onmessage = (event) => {
            this.handleMessage(event);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            UI.showError('연결 오류가 발생했습니다.');
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            AppState.isConnected = false;
            UI.updateConnectionStatus('disconnected');

            // Auto-reconnect logic
            if (AppState.isInterviewActive && this.reconnectAttempts < Config.reconnectAttempts) {
                this.reconnectAttempts++;
                console.log(`Reconnecting... (${this.reconnectAttempts}/${Config.reconnectAttempts})`);
                setTimeout(() => this.connect(sessionId), Config.reconnectDelay);
            }
        };

        AppState.wsConnection = this.ws;
    },

    handleMessage(event) {
        // Check if binary data (video frame)
        if (event.data instanceof ArrayBuffer) {
            this.handleBinaryMessage(event.data);
            return;
        }

        // JSON message
        try {
            const data = JSON.parse(event.data);
            this.handleJsonMessage(data);
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    },

    handleBinaryMessage(data) {
        // Binary data is a video frame (JPEG)
        const blob = new Blob([data], { type: 'image/jpeg' });
        const imageUrl = URL.createObjectURL(blob);

        AvatarRenderer.renderFrame(imageUrl);
    },

    handleJsonMessage(data) {
        console.log('WebSocket message:', data.type, data);

        switch (data.type) {
            case 'connected':
                console.log('Session connected:', data.session_id);
                console.log('Server capabilities:', data.capabilities);
                break;

            case 'greeting':
                // Initial greeting from avatar
                if (data.text) {
                    UI.addTranscript('interviewer', data.text);
                }
                if (data.audio) {
                    AudioPlayer.playBase64Audio(data.audio);
                }
                break;

            case 'response':
                // AI response (legacy)
                if (data.text) {
                    UI.addTranscript('interviewer', data.text);
                }
                if (data.audio) {
                    AudioPlayer.playBase64Audio(data.audio);
                }
                if (data.metrics) {
                    UI.updateLatency(data.metrics.total_ms);
                }
                break;

            case 'stt_result':
                // STT 결과 (Whisper)
                if (data.is_final) {
                    UI.addTranscript('candidate', data.text);
                    UI.clearLiveSubtitle();
                } else {
                    UI.updateLiveSubtitle(data.text);
                }
                break;

            case 'llm_response':
                // LLM 응답 (Pipecat)
                if (data.text) {
                    UI.addTranscript('interviewer', data.text);
                }
                break;

            case 'tts_audio':
                // TTS 오디오 (Pipecat - PCM format)
                if (data.audio) {
                    AudioPlayer.playPcmAudio(data.audio, data.sample_rate || 24000);
                }
                break;

            case 'transcript':
                // User speech transcript (legacy)
                if (data.is_final) {
                    UI.addTranscript('candidate', data.text);
                    UI.clearLiveSubtitle();
                } else {
                    UI.updateLiveSubtitle(data.text);
                }
                break;

            case 'status_update':
                UI.updateInterviewStatus(data);
                break;

            case 'stage_change':
                AppState.interviewStage = data.stage;
                UI.updateInterviewStage(data.stage);
                break;

            case 'state':
                // Avatar state update (legacy)
                this.handleStateChange(data.state);
                break;

            case 'state_change':
                // 파이프라인 상태 변경 (Pipecat)
                this.handleStateChange(data.state);
                break;

            case 'video_frame':
                // Video frame as base64
                if (data.frame) {
                    AvatarRenderer.renderBase64Frame(data.frame);
                }
                break;

            case 'pong':
                // Ping response
                break;

            case 'reset_complete':
                console.log('Conversation reset complete');
                break;

            case 'error':
                console.error('Server error:', data.message);
                UI.showError(data.message);
                break;

            default:
                console.warn('Unknown message type:', data.type);
        }
    },

    handleStateChange(state) {
        UI.setIndicator('listening', state === 'listening');
        UI.setIndicator('processing', state === 'processing');
        UI.setIndicator('speaking', state === 'speaking');

        if (state === 'speaking') {
            UI.showAvatarSpeaking(true);
        } else {
            UI.showAvatarSpeaking(false);
        }
    },

    sendAudioData(audioData) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            // Send as binary
            this.ws.send(audioData);
        }
    },

    sendTextMessage(text) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'text_input',
                text: text
            }));
        }
    },

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            console.error('WebSocket not connected');
        }
    },

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
            AppState.wsConnection = null;
            AppState.isConnected = false;
        }
    }
};

// ============================================================================
// Audio Capture
// ============================================================================

const AudioCapture = {
    async initialize() {
        try {
            // Request microphone access
            AppState.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: Config.sampleRate,
                    channelCount: Config.channels,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                }
            });

            // Create audio context
            AppState.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: Config.sampleRate
            });

            // Create source from microphone
            const source = AppState.audioContext.createMediaStreamSource(AppState.mediaStream);

            // Create analyser for volume meter
            AppState.analyser = AppState.audioContext.createAnalyser();
            AppState.analyser.fftSize = 256;
            source.connect(AppState.analyser);

            // Create script processor for audio data
            AppState.audioProcessor = AppState.audioContext.createScriptProcessor(
                Config.bufferSize,
                Config.channels,
                Config.channels
            );

            AppState.audioProcessor.onaudioprocess = (event) => {
                if (AppState.isMicEnabled && AppState.isConnected) {
                    const inputData = event.inputBuffer.getChannelData(0);
                    this.sendAudioChunk(inputData);
                }
            };

            source.connect(AppState.audioProcessor);
            AppState.audioProcessor.connect(AppState.audioContext.destination);

            // Start volume monitoring
            this.startVolumeMonitoring();

            console.log('Audio capture initialized');
            return true;

        } catch (error) {
            console.error('Failed to initialize audio capture:', error);
            throw error;
        }
    },

    sendAudioChunk(float32Array) {
        // Convert Float32Array to Int16Array (PCM 16-bit)
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Send as binary data
        WebSocketManager.sendAudioData(int16Array.buffer);
    },

    startVolumeMonitoring() {
        if (AppState.volumeCheckInterval) return;

        const dataArray = new Uint8Array(AppState.analyser.frequencyBinCount);

        AppState.volumeCheckInterval = setInterval(() => {
            if (!AppState.analyser) return;

            AppState.analyser.getByteFrequencyData(dataArray);

            // Calculate average volume
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                sum += dataArray[i];
            }
            const average = sum / dataArray.length;

            // Update UI (0-100 scale)
            const volumePercent = Math.min(100, (average / 128) * 100);
            UI.updateVolumeLevel(volumePercent);

        }, 50);
    },

    stopVolumeMonitoring() {
        if (AppState.volumeCheckInterval) {
            clearInterval(AppState.volumeCheckInterval);
            AppState.volumeCheckInterval = null;
            UI.updateVolumeLevel(0);
        }
    },

    enableMicrophone() {
        AppState.isMicEnabled = true;
        if (AppState.audioContext && AppState.audioContext.state === 'suspended') {
            AppState.audioContext.resume();
        }
    },

    disableMicrophone() {
        AppState.isMicEnabled = false;
    },

    cleanup() {
        this.stopVolumeMonitoring();

        if (AppState.audioProcessor) {
            AppState.audioProcessor.disconnect();
            AppState.audioProcessor = null;
        }

        if (AppState.audioContext) {
            AppState.audioContext.close();
            AppState.audioContext = null;
        }

        if (AppState.mediaStream) {
            AppState.mediaStream.getTracks().forEach(track => track.stop());
            AppState.mediaStream = null;
        }

        AppState.analyser = null;
    }
};

// ============================================================================
// Audio Player
// ============================================================================

const AudioPlayer = {
    audioElement: null,
    audioContext: null,
    pcmQueue: [],
    isPlayingPcm: false,

    initialize() {
        this.audioElement = document.getElementById('audioPlayer');

        this.audioElement.addEventListener('ended', () => {
            AppState.isPlayingAudio = false;
            this.playNextInQueue();
        });

        this.audioElement.addEventListener('error', (e) => {
            console.error('Audio playback error:', e);
            AppState.isPlayingAudio = false;
            this.playNextInQueue();
        });

        // Initialize Web Audio API for PCM playback
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
            console.error('Failed to create AudioContext:', e);
        }
    },

    playBase64Audio(base64Data) {
        // Add to queue
        AppState.audioQueue.push(base64Data);

        // Start playing if not already
        if (!AppState.isPlayingAudio) {
            this.playNextInQueue();
        }
    },

    playNextInQueue() {
        if (AppState.audioQueue.length === 0) {
            AppState.isPlayingAudio = false;
            return;
        }

        AppState.isPlayingAudio = true;
        const base64Data = AppState.audioQueue.shift();

        // Create audio URL from base64
        const audioUrl = `data:audio/mp3;base64,${base64Data}`;
        this.audioElement.src = audioUrl;
        this.audioElement.play().catch(error => {
            console.error('Failed to play audio:', error);
            AppState.isPlayingAudio = false;
            this.playNextInQueue();
        });
    },

    // PCM 오디오 재생 (Pipecat TTS용)
    playPcmAudio(base64Data, sampleRate = 24000) {
        // Add to PCM queue
        this.pcmQueue.push({ data: base64Data, sampleRate });

        // Start playing if not already
        if (!this.isPlayingPcm) {
            this.playNextPcm();
        }
    },

    async playNextPcm() {
        if (this.pcmQueue.length === 0) {
            this.isPlayingPcm = false;
            UI.showAvatarSpeaking(false);
            return;
        }

        this.isPlayingPcm = true;
        UI.showAvatarSpeaking(true);

        const { data, sampleRate } = this.pcmQueue.shift();

        try {
            // Ensure AudioContext is running (may be suspended due to autoplay policy)
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            // Decode base64 to ArrayBuffer
            const binaryString = atob(data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }

            // Convert PCM int16 to float32
            const pcmData = new Int16Array(bytes.buffer);
            const floatData = new Float32Array(pcmData.length);
            for (let i = 0; i < pcmData.length; i++) {
                floatData[i] = pcmData[i] / 32768.0;
            }

            // Create AudioBuffer
            const audioBuffer = this.audioContext.createBuffer(1, floatData.length, sampleRate);
            audioBuffer.getChannelData(0).set(floatData);

            // Create and play source
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);

            source.onended = () => {
                this.playNextPcm();
            };

            source.start();

        } catch (error) {
            console.error('Failed to play PCM audio:', error);
            this.playNextPcm();
        }
    },

    stop() {
        AppState.audioQueue = [];
        this.pcmQueue = [];
        AppState.isPlayingAudio = false;
        this.isPlayingPcm = false;
        if (this.audioElement) {
            this.audioElement.pause();
            this.audioElement.currentTime = 0;
        }
        UI.showAvatarSpeaking(false);
    }
};

// ============================================================================
// Avatar Renderer
// ============================================================================

const AvatarRenderer = {
    initialize() {
        AppState.avatarCanvas = document.getElementById('avatarCanvas');
        AppState.avatarCtx = AppState.avatarCanvas.getContext('2d');

        // Set canvas size
        AppState.avatarCanvas.width = Config.frameWidth;
        AppState.avatarCanvas.height = Config.frameHeight;

        // Initial placeholder
        this.drawPlaceholder();
    },

    drawPlaceholder() {
        const ctx = AppState.avatarCtx;
        const canvas = AppState.avatarCanvas;

        // Draw gradient background
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
        gradient.addColorStop(0, '#667eea');
        gradient.addColorStop(1, '#764ba2');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    },

    renderFrame(imageUrl) {
        const img = new Image();
        img.onload = () => {
            AppState.avatarCtx.drawImage(img, 0, 0, Config.frameWidth, Config.frameHeight);
            URL.revokeObjectURL(imageUrl);

            // Hide placeholder
            document.getElementById('avatarPlaceholder').classList.add('hidden');
        };
        img.onerror = () => {
            console.error('Failed to load frame image');
            URL.revokeObjectURL(imageUrl);
        };
        img.src = imageUrl;
    },

    renderBase64Frame(base64Data) {
        const img = new Image();
        img.onload = () => {
            AppState.avatarCtx.drawImage(img, 0, 0, Config.frameWidth, Config.frameHeight);

            // Hide placeholder
            document.getElementById('avatarPlaceholder').classList.add('hidden');
        };
        img.src = `data:image/jpeg;base64,${base64Data}`;
    },

    showPlaceholder() {
        document.getElementById('avatarPlaceholder').classList.remove('hidden');
        this.drawPlaceholder();
    }
};

// ============================================================================
// UI Layer
// ============================================================================

const UI = {
    elements: {},

    initialize() {
        // Cache DOM elements
        this.elements = {
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            micToggle: document.getElementById('micToggle'),
            micIcon: document.getElementById('micIcon'),
            micText: document.getElementById('micText'),
            themeToggle: document.getElementById('themeToggle'),

            connectionStatus: document.getElementById('connectionStatus'),
            listeningStatus: document.getElementById('listeningStatus'),
            processingStatus: document.getElementById('processingStatus'),
            speakingStatus: document.getElementById('speakingStatus'),

            interviewStage: document.getElementById('interviewStage'),
            questionCount: document.getElementById('questionCount'),
            elapsedTime: document.getElementById('elapsedTime'),
            latencyDisplay: document.getElementById('latencyDisplay'),

            volumeLevel: document.getElementById('volumeLevel'),
            transcriptContainer: document.getElementById('transcriptContainer'),
            liveSubtitle: document.getElementById('liveSubtitle'),

            textInput: document.getElementById('textInput'),
            sendTextBtn: document.getElementById('sendTextBtn'),

            avatarPlaceholder: document.getElementById('avatarPlaceholder'),
            avatarSpeakingIndicator: document.getElementById('avatarSpeakingIndicator'),

            loadingOverlay: document.getElementById('loadingOverlay'),
            errorModal: document.getElementById('errorModal'),
            errorMessage: document.getElementById('errorMessage'),
            errorClose: document.getElementById('errorClose'),
        };

        // Set up event listeners
        this.setupEventListeners();

        // Load theme preference
        this.loadTheme();

        // Initialize components
        AvatarRenderer.initialize();
        AudioPlayer.initialize();
    },

    setupEventListeners() {
        this.elements.startBtn.addEventListener('click', () => App.startInterview());
        this.elements.stopBtn.addEventListener('click', () => App.stopInterview());
        this.elements.micToggle.addEventListener('click', () => App.toggleMicrophone());
        this.elements.themeToggle.addEventListener('click', () => this.toggleTheme());
        this.elements.errorClose.addEventListener('click', () => this.hideError());

        // Text input
        this.elements.sendTextBtn.addEventListener('click', () => this.sendTextInput());
        this.elements.textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendTextInput();
            }
        });
    },

    sendTextInput() {
        const text = this.elements.textInput.value.trim();
        if (text && AppState.isConnected) {
            WebSocketManager.sendTextMessage(text);
            this.addTranscript('candidate', text);
            this.elements.textInput.value = '';
        }
    },

    showLoading(text = '연결 중...') {
        this.elements.loadingOverlay.querySelector('.loading-text').textContent = text;
        this.elements.loadingOverlay.classList.remove('hidden');
    },

    hideLoading() {
        this.elements.loadingOverlay.classList.add('hidden');
    },

    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.errorModal.classList.remove('hidden');
    },

    hideError() {
        this.elements.errorModal.classList.add('hidden');
    },

    updateConnectionStatus(status) {
        const statusMap = {
            'idle': '연결 대기 중',
            'connecting': '연결 중...',
            'connected': '연결됨',
            'disconnected': '연결 끊김',
        };

        this.elements.connectionStatus.textContent = statusMap[status] || status;
        this.elements.connectionStatus.className = `status-badge status-${status}`;
    },

    setIndicator(indicator, active) {
        const elementMap = {
            'listening': this.elements.listeningStatus,
            'processing': this.elements.processingStatus,
            'speaking': this.elements.speakingStatus,
        };

        const element = elementMap[indicator];
        if (element) {
            element.classList.toggle('active', active);
        }
    },

    showAvatarSpeaking(show) {
        this.elements.avatarSpeakingIndicator.classList.toggle('hidden', !show);
    },

    updateInterviewStage(stage) {
        const stageMap = {
            'idle': '대기 중',
            'greeting': '인사',
            'self_introduction': '자기소개',
            'experience': '경력 질문',
            'technical': '기술 질문',
            'situational': '상황 질문',
            'closing': '마무리',
            'farewell': '종료',
        };

        this.elements.interviewStage.textContent = stageMap[stage] || stage;
    },

    updateQuestionCount(count) {
        this.elements.questionCount.textContent = count;
    },

    updateLatency(ms) {
        const latencyBadge = this.elements.latencyDisplay;
        const valueSpan = latencyBadge.querySelector('.latency-value');
        valueSpan.textContent = `${Math.round(ms)} ms`;

        // Color based on latency
        latencyBadge.classList.remove('latency-good', 'latency-medium', 'latency-poor');
        if (ms < 500) {
            latencyBadge.classList.add('latency-good');
        } else if (ms < 1000) {
            latencyBadge.classList.add('latency-medium');
        } else {
            latencyBadge.classList.add('latency-poor');
        }
    },

    updateElapsedTime() {
        if (!AppState.startTime) return;

        const elapsed = Math.floor((Date.now() - AppState.startTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;

        this.elements.elapsedTime.textContent =
            `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    },

    startElapsedTimer() {
        AppState.startTime = Date.now();
        AppState.elapsedTimeInterval = setInterval(() => this.updateElapsedTime(), 1000);
    },

    stopElapsedTimer() {
        if (AppState.elapsedTimeInterval) {
            clearInterval(AppState.elapsedTimeInterval);
            AppState.elapsedTimeInterval = null;
        }
    },

    updateVolumeLevel(level) {
        this.elements.volumeLevel.style.width = `${Math.min(level, 100)}%`;
    },

    addTranscript(speaker, text) {
        // Remove placeholder if exists
        const placeholder = this.elements.transcriptContainer.querySelector('.transcript-placeholder');
        if (placeholder) {
            placeholder.remove();
        }

        const transcriptItem = document.createElement('div');
        transcriptItem.className = `transcript-item transcript-${speaker}`;

        const timestamp = new Date().toLocaleTimeString('ko-KR', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });

        const speakerLabel = speaker === 'interviewer' ? '면접관' : '지원자';

        transcriptItem.innerHTML = `
            <div class="transcript-header">
                <span class="transcript-speaker">${speakerLabel}</span>
                <span class="transcript-time">${timestamp}</span>
            </div>
            <div class="transcript-text">${text}</div>
        `;

        this.elements.transcriptContainer.appendChild(transcriptItem);
        this.elements.transcriptContainer.scrollTop = this.elements.transcriptContainer.scrollHeight;
    },

    updateLiveSubtitle(text) {
        this.elements.liveSubtitle.textContent = text;
        this.elements.liveSubtitle.classList.add('active');
    },

    clearLiveSubtitle() {
        this.elements.liveSubtitle.textContent = '';
        this.elements.liveSubtitle.classList.remove('active');
    },

    updateInterviewStatus(data) {
        if (data.stage) {
            this.updateInterviewStage(data.stage);
        }
        if (data.question_count !== undefined) {
            this.updateQuestionCount(data.question_count);
        }
    },

    enableControls() {
        this.elements.stopBtn.disabled = false;
        this.elements.micToggle.disabled = false;
        this.elements.startBtn.disabled = true;
        this.elements.textInput.disabled = false;
        this.elements.sendTextBtn.disabled = false;
    },

    disableControls() {
        this.elements.stopBtn.disabled = true;
        this.elements.micToggle.disabled = true;
        this.elements.startBtn.disabled = false;
        this.elements.textInput.disabled = true;
        this.elements.sendTextBtn.disabled = true;
    },

    updateMicButton(enabled) {
        this.elements.micIcon.textContent = enabled ? '🎤' : '🔇';
        this.elements.micText.textContent = enabled ? '마이크 끄기' : '마이크 켜기';
        this.elements.micToggle.classList.toggle('active', enabled);
    },

    toggleTheme() {
        AppState.theme = AppState.theme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', AppState.theme);
        localStorage.setItem('theme', AppState.theme);
    },

    loadTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        AppState.theme = savedTheme;
        document.documentElement.setAttribute('data-theme', savedTheme);
    },

    reset() {
        this.updateConnectionStatus('idle');
        this.updateInterviewStage('idle');
        this.updateQuestionCount(0);
        this.stopElapsedTimer();
        this.elements.elapsedTime.textContent = '00:00';
        this.updateVolumeLevel(0);
        this.clearLiveSubtitle();

        // Reset latency display
        const valueSpan = this.elements.latencyDisplay.querySelector('.latency-value');
        valueSpan.textContent = '-- ms';
        this.elements.latencyDisplay.classList.remove('latency-good', 'latency-medium', 'latency-poor');

        // Clear transcript
        this.elements.transcriptContainer.innerHTML = `
            <div class="transcript-placeholder">
                면접이 시작되면 대화 내용이 여기에 표시됩니다.
            </div>
        `;

        // Reset indicators
        this.setIndicator('listening', false);
        this.setIndicator('processing', false);
        this.setIndicator('speaking', false);

        // Show avatar placeholder
        AvatarRenderer.showPlaceholder();
    }
};

// ============================================================================
// Main Application
// ============================================================================

const App = {
    async initialize() {
        UI.initialize();
        console.log('Application initialized');
    },

    async startInterview() {
        try {
            UI.showLoading('면접 세션 생성 중...');

            // Start interview via API
            const response = await API.startInterview();
            AppState.sessionId = response.session_id;

            console.log('Interview session created:', AppState.sessionId);

            // Initialize audio capture
            UI.showLoading('마이크 권한 요청 중...');
            await AudioCapture.initialize();

            // Connect to WebSocket
            UI.showLoading('실시간 연결 중...');
            WebSocketManager.connect(AppState.sessionId);

            // Wait for WebSocket connection
            await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => reject(new Error('Connection timeout')), 10000);
                const checkConnection = setInterval(() => {
                    if (AppState.isConnected) {
                        clearInterval(checkConnection);
                        clearTimeout(timeout);
                        resolve();
                    }
                }, 100);
            });

            // Update state
            AppState.isInterviewActive = true;

            // Update UI
            UI.hideLoading();
            UI.enableControls();
            UI.startElapsedTimer();
            UI.updateConnectionStatus('connected');

            console.log('Interview started successfully');

        } catch (error) {
            console.error('Failed to start interview:', error);
            UI.hideLoading();
            UI.showError('면접 시작에 실패했습니다: ' + error.message);
            await this.cleanup();
        }
    },

    async stopInterview() {
        try {
            if (!AppState.sessionId) return;

            UI.showLoading('면접 종료 중...');

            // End interview via API
            await API.endInterview(AppState.sessionId);

            // Cleanup
            await this.cleanup();

            UI.hideLoading();
            alert('면접이 종료되었습니다. 수고하셨습니다!');

        } catch (error) {
            console.error('Failed to stop interview:', error);
            UI.hideLoading();
            UI.showError('면접 종료 중 오류가 발생했습니다: ' + error.message);
        }
    },

    async toggleMicrophone() {
        try {
            if (AppState.isMicEnabled) {
                AudioCapture.disableMicrophone();
                UI.updateMicButton(false);
            } else {
                AudioCapture.enableMicrophone();
                UI.updateMicButton(true);
            }
        } catch (error) {
            console.error('Failed to toggle microphone:', error);
            UI.showError('마이크 설정 변경에 실패했습니다: ' + error.message);
        }
    },

    async cleanup() {
        // Stop audio playback
        AudioPlayer.stop();

        // Cleanup audio capture
        AudioCapture.cleanup();

        // Disconnect WebSocket
        WebSocketManager.disconnect();

        // Reset state
        AppState.sessionId = null;
        AppState.isInterviewActive = false;
        AppState.isMicEnabled = false;
        AppState.interviewStage = 'idle';
        AppState.questionCount = 0;
        AppState.startTime = null;

        // Reset UI
        UI.disableControls();
        UI.updateMicButton(false);
        UI.reset();
    }
};

// ============================================================================
// Initialize on load
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    App.initialize();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (AppState.isInterviewActive) {
        App.cleanup();
    }
});
