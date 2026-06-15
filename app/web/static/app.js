(() => {
  /* ─── CONFIG & STATE ─── */
  const appConfig = window.APP_CONFIG || {};
  const SESSION_KEY = "ai_realtime_translation_session_id";

  const state = {
    sessionId: localStorage.getItem(SESSION_KEY) || crypto.randomUUID(),
    activeMode: "sign",
    signStream: null,
    signCapturePending: false,
    signCaptureTimer: null,
    signCaptureIntervalMs: Number(appConfig.sign_capture_interval_ms || 350),
    recordingSeconds: Number(appConfig.recording_seconds || 5),
    audioStream: null,
    mediaRecorder: null,
    latestHistory: [],
    /* overlay state — updated from server response */
    handLandmarks: null,   // [{x,y,z}, …] normalised 0..1
    animFrame: null,
  };

  localStorage.setItem(SESSION_KEY, state.sessionId);

  /* ─── ELEMENT REFS ─── */
  const $ = (id) => document.getElementById(id);

  const els = {
    /* nav */
    sessionLabel:    $("sessionLabel"),
    modelStatusPill: $("modelStatusPill"),
    /* tabs */
    tabSign:  $("tabSign"),
    tabText:  $("tabText"),
    tabAudio: $("tabAudio"),
    /* panels */
    modeSign:  $("modeSign"),
    modeText:  $("modeText"),
    modeAudio: $("modeAudio"),
    /* camera */
    cameraWrap:    $("cameraWrap"),
    cameraFeed:    $("cameraFeed"),
    overlayCanvas: $("overlayCanvas"),
    cameraPrompt:  $("cameraPrompt"),
    /* badges */
    handBadge:   $("handBadge"),
    handLabel:   $("handLabel"),
    detectBadge: $("detectBadge"),
    detectLabel: $("detectLabel"),
    /* result bar */
    gestureCandidate: $("gestureCandidate"),
    confidenceValue:  $("confidenceValue"),
    confBarFill:      $("confBarFill"),
    detectedText:     $("detectedText"),
    /* status */
    statusMsg:      $("statusMsg"),
    textStatusMsg:  $("textStatusMsg"),
    audioStatusMsg: $("audioStatusMsg"),
    /* side panel */
    modelInfo:       $("modelInfo"),
    historyList:     $("historyList"),
    clearHistoryBtn: $("clearHistoryBtn"),
    audioPlayer:     $("audioPlayer"),
    /* text mode */
    textInput:       $("textInput"),
    speakTextBtn:    $("speakTextBtn"),
    showSignBtn:     $("showSignBtn"),
    textOutputCard:  $("textOutputCard"),
    signBackend:     $("signBackend"),
    signGloss:       $("signGloss"),
    modelVideo:      $("modelVideo"),
    signStoryboard:  $("signStoryboard"),
    textAudioPlayer: $("textAudioPlayer"),
    /* audio mode */
    languageSelect:    $("languageSelect"),
    audioFile:         $("audioFile"),
    recordAudioBtn:    $("recordAudioBtn"),
    uploadAudioBtn:    $("uploadAudioBtn"),
    audioOutputCard:   $("audioOutputCard"),
    audioTranscript:   $("audioTranscript"),
    audioResultPlayer: $("audioResultPlayer"),
  };

  if (appConfig.language) {
    els.languageSelect.value = appConfig.language;
  }

  /* ─── UTILS ─── */
  const esc = (v) =>
    String(v || "")
      .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;").replaceAll('"', "&quot;");

  const setStatus = (msg, isError = false) => {
    els.statusMsg.textContent = msg;
    els.statusMsg.classList.toggle("err", isError);
  };
  const setTextStatus  = (msg) => { els.textStatusMsg.textContent  = msg; };
  const setAudioStatus = (msg) => { els.audioStatusMsg.textContent = msg; };

  const playAudio = (url, player) => {
    if (!url || !player) return;
    player.src = url;
    player.load();
    player.play().catch(() => {});
  };

  /* ─── HAND DETECTION VISUAL FEEDBACK ─── */
  const setHandState = (detected) => {
    els.handBadge.classList.toggle("hand-on", detected);
    els.handLabel.textContent = detected ? "Hand detected" : "No hand";
  };

  const setDetectingState = (active) => {
    els.detectBadge.classList.toggle("active", active);
    els.detectLabel.textContent = active ? "Recognizing…" : "Idle";
  };

  /* ─── HAND SKELETON CONNECTIONS (MediaPipe 21-pt model) ─── */
  const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],
  ];
  const FINGERTIP_IDS = [4, 8, 12, 16, 20];

  /* ─── CANVAS OVERLAY RENDER LOOP ─── */
  const syncCanvasSize = () => {
    const vid = els.cameraFeed;
    const c   = els.overlayCanvas;
    const W   = vid.videoWidth  || els.cameraWrap.clientWidth;
    const H   = vid.videoHeight || els.cameraWrap.clientHeight;
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  };

  const drawOverlay = () => {
    syncCanvasSize();
    const canvas = els.overlayCanvas;
    const ctx    = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const lms = state.handLandmarks;
    if (lms && lms.length === 21) {
      /* skeleton lines */
      ctx.strokeStyle = "rgba(63,185,80,0.85)";
      ctx.lineWidth   = 2.5;
      ctx.globalAlpha = 0.85;
      ctx.lineCap     = "round";
      for (const [i, j] of HAND_CONNECTIONS) {
        const a = lms[i], b = lms[j];
        if (!a || !b) continue;
        ctx.beginPath();
        ctx.moveTo(a.x * W, a.y * H);
        ctx.lineTo(b.x * W, b.y * H);
        ctx.stroke();
      }

      /* regular joint dots */
      ctx.globalAlpha = 1;
      for (let k = 0; k < 21; k++) {
        if (FINGERTIP_IDS.includes(k)) continue; // drawn separately
        const lm = lms[k];
        ctx.beginPath();
        ctx.arc(lm.x * W, lm.y * H, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#3fb950";
        ctx.fill();
      }

      /* fingertip highlights */
      for (const t of FINGERTIP_IDS) {
        const lm = lms[t];
        /* outer ring */
        ctx.beginPath();
        ctx.arc(lm.x * W, lm.y * H, 8, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(88,166,255,0.6)";
        ctx.lineWidth   = 2;
        ctx.stroke();
        /* inner dot */
        ctx.beginPath();
        ctx.arc(lm.x * W, lm.y * H, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = "#58a6ff";
        ctx.fill();
      }

      /* wrist label */
      const wrist = lms[0];
      ctx.font      = "bold 13px Inter, sans-serif";
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.fillText("✋", wrist.x * W - 10, wrist.y * H + 20);
    }

    state.animFrame = requestAnimationFrame(drawOverlay);
  };

  const startOverlay = () => {
    if (state.animFrame) cancelAnimationFrame(state.animFrame);
    drawOverlay();
  };

  const stopOverlay = () => {
    if (state.animFrame) { cancelAnimationFrame(state.animFrame); state.animFrame = null; }
    const ctx = els.overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, els.overlayCanvas.width, els.overlayCanvas.height);
  };

  /* ─── CAMERA ─── */
  const startCamera = async () => {
    if (state.signStream) return;
    try {
      state.signStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: appConfig.camera_facing_mode || "user",
          width:  { ideal: 1280 },
          height: { ideal: 720  },
        },
        audio: false,
      });
      els.cameraFeed.srcObject = state.signStream;

      await new Promise((resolve) => {
        els.cameraFeed.onloadedmetadata = resolve;
      });

      /* sync canvas once video dimensions are known */
      syncCanvasSize();

      /* hide the "camera starting" prompt */
      els.cameraPrompt.classList.add("gone");
      setTimeout(() => { els.cameraPrompt.style.display = "none"; }, 420);

      els.sessionLabel.textContent = `Session ${state.sessionId.slice(0, 8)}`;
      setStatus("Camera active — show a hand sign");
      setDetectingState(true);

      /* start overlay render loop */
      startOverlay();
      /* start capture → server loop */
      startCaptureLoop();

    } catch (err) {
      setStatus(`Camera error: ${err.message}`, true);
    }
  };

  const stopCamera = () => {
    if (state.signCaptureTimer) { clearTimeout(state.signCaptureTimer); state.signCaptureTimer = null; }
    stopOverlay();
    if (state.signStream) {
      state.signStream.getTracks().forEach((t) => t.stop());
      state.signStream = null;
    }
    els.cameraFeed.srcObject = null;
    state.handLandmarks = null;
    setDetectingState(false);
    setHandState(false);
  };

  /* ─── CAPTURE LOOP (sends frames to server for recognition) ─── */
  const startCaptureLoop = () => {
    if (state.activeMode !== "sign" || !state.signStream) return;
    state.signCaptureTimer = setTimeout(async () => {
      await captureSignFrame();
      startCaptureLoop();
    }, state.signCaptureIntervalMs);
  };

  const captureFrameBlob = async () => {
    const video = els.cameraFeed;
    if (!video.videoWidth || !video.videoHeight) return null;
    const canvas = document.createElement("canvas");
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    return new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.82));
  };

  const captureSignFrame = async () => {
    if (state.signCapturePending) return;
    const blob = await captureFrameBlob();
    if (!blob) return;

    state.signCapturePending = true;
    const form = new FormData();
    form.append("session_id", state.sessionId);
    form.append("frame", blob, "frame.jpg");

    try {
      const res  = await fetch("/api/sign/recognize", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Recognition failed");

      /* ── update server-provided hand landmarks for canvas overlay ── */
      if (Array.isArray(data.hand_landmarks) && data.hand_landmarks.length === 21) {
        state.handLandmarks = data.hand_landmarks;
      } else {
        state.handLandmarks = null;
      }

      /* ── visual feedback ── */
      setHandState(data.has_hand === true);

      /* ── gesture display ── */
      const candidate  = data.candidate  || "";
      const confidence = Number(data.confidence || 0);
      els.gestureCandidate.textContent = candidate || "—";
      els.confidenceValue.textContent  = confidence > 0 ? `${(confidence * 100).toFixed(0)}%` : "0%";
      els.confBarFill.style.width      = `${Math.min(100, confidence * 100).toFixed(1)}%`;
      els.detectedText.textContent     = data.transcript || data.candidate || "Waiting for gestures…";

      setStatus(data.message || "Frame processed");
      updateHistory(data.history);
      renderModelInfo(data.model_stats || {});

      if (data.audio_url) playAudio(data.audio_url, els.audioPlayer);

    } catch (err) {
      setStatus(err.message, true);
    } finally {
      state.signCapturePending = false;
    }
  };

  /* ─── MODEL INFO ─── */
  const renderModelInfo = (stats) => {
    const trained  = Boolean(stats.trained);
    const labels   = Array.isArray(stats.labels) ? stats.labels : [];
    const accuracy = typeof stats.validation_accuracy === "number"
      ? `${(stats.validation_accuracy * 100).toFixed(1)}%` : "n/a";
    const trainedAt = stats.trained_at || "n/a";

    els.modelStatusPill.textContent = `Model: ${trained ? "Loaded ✓" : "Missing"}`;
    els.modelStatusPill.classList.toggle("loaded",  trained);
    els.modelStatusPill.classList.toggle("missing", !trained);

    els.modelInfo.innerHTML = `
      <div class="info-row"><strong>Status:</strong> ${trained ? "✅ Trained model loaded" : "⚠️ Model not found"}</div>
      <div class="info-row"><strong>Labels:</strong> ${labels.length} signs</div>
      <div class="info-row"><strong>Accuracy:</strong> ${esc(accuracy)}</div>
      <div class="info-row"><strong>Trained:</strong> ${esc(trainedAt)}</div>
      ${stats.load_error
        ? `<div class="info-row" style="color:#f85149;margin-top:6px;font-size:0.72rem;">${esc(stats.load_error)}</div>`
        : ""}
    `;
  };

  /* ─── HISTORY ─── */
  const updateHistory = (history) => {
    state.latestHistory = history || [];
    if (!state.latestHistory.length) {
      els.historyList.innerHTML = `<div class="history-empty">No translations yet.</div>`;
      return;
    }
    els.historyList.innerHTML = [...state.latestHistory]
      .reverse().slice(0, 12)
      .map((item) => `
        <div class="history-item">
          <div class="hi-mode">${esc(item.mode)} · ${esc(item.timestamp || "")}</div>
          <div class="hi-in">${esc(item.input_text || "")}</div>
          ${item.output_text ? `<div class="hi-out">${esc(item.output_text)}</div>` : ""}
        </div>`)
      .join("");
  };

  const clearHistory = async () => {
    try {
      await fetch("/api/history/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId }),
      });
      updateHistory([]);
      els.detectedText.textContent = "History cleared.";
      setStatus("Session history cleared");
    } catch (err) { setStatus(err.message, true); }
  };

  const loadHistory = async () => {
    try {
      const res  = await fetch(`/api/history?session_id=${encodeURIComponent(state.sessionId)}`);
      const data = await res.json();
      updateHistory(data.history || []);
      if (data.transcript) els.detectedText.textContent = data.transcript;
    } catch (_) {}
  };

  /* ─── TEXT → SIGN ─── */
  const speakText = async () => {
    const text = els.textInput.value.trim();
    if (!text) { setTextStatus("Please type some text first."); return; }
    setTextStatus("Generating speech…");
    els.speakTextBtn.disabled = true;
    try {
      const res  = await fetch("/api/text/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId, text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "TTS failed");
      setTextStatus("Speech generated ✓");
      updateHistory(data.history);
      if (data.audio_url) playAudio(data.audio_url, els.textAudioPlayer);
    } catch (err) {
      setTextStatus(`Error: ${err.message}`);
    } finally {
      els.speakTextBtn.disabled = false;
    }
  };

  const showSign = async () => {
    const text = els.textInput.value.trim();
    if (!text) { setTextStatus("Please type some text first."); return; }
    setTextStatus("Generating sign playback…");
    els.showSignBtn.disabled = true;
    try {
      const res  = await fetch("/api/text/sign", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId, text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Sign failed");

      els.signBackend.textContent   = `Backend: ${data.backend || "storyboard"}`;
      els.signGloss.textContent     = data.gloss ? `Gloss: ${data.gloss}` : "Gloss: n/a";
      els.textOutputCard.style.display = "";
      setTextStatus(`Sign output ready via ${data.backend || "storyboard"}`);
      updateHistory(data.history);

      if (data.video_url) {
        els.modelVideo.classList.remove("hidden");
        els.signStoryboard.innerHTML = "";
        els.modelVideo.src = data.video_url;
        els.modelVideo.load();
        els.modelVideo.play().catch(() => {});
      } else if (data.steps && data.steps.length) {
        els.modelVideo.classList.add("hidden");
        els.modelVideo.src = ""; // Clear source to avoid 404s or stale video
        els.signStoryboard.innerHTML = data.steps
          .map((s, i) => `
            <div class="story-step">
              <strong>${i + 1}. ${esc(s.label || `step ${i + 1}`)}</strong>
              <span>${esc(s.description || "")}</span>
            </div>`)
          .join("");
      } else {
        els.modelVideo.classList.add("hidden");
        els.modelVideo.src = "";
        els.signStoryboard.innerHTML = `<div class="history-empty">No sign storyboard available.</div>`;
      }
    } catch (err) {
      setTextStatus(`Error: ${err.message}`);
    } finally {
      els.showSignBtn.disabled = false;
    }
  };

  /* ─── AUDIO INPUT ─── */
  const sendAudioBlob = async (blob, filename) => {
    const form = new FormData();
    form.append("session_id", state.sessionId);
    form.append("audio", blob, filename);
    form.append("language", els.languageSelect.value);
    try {
      const res  = await fetch("/api/audio/transcribe", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Transcription failed");
      const result = data.output_text || data.raw_text || "No transcript";
      els.audioTranscript.textContent   = result;
      els.audioOutputCard.style.display = "";
      setAudioStatus("Transcription complete ✓");
      updateHistory(data.history);
      if (data.audio_url) playAudio(data.audio_url, els.audioResultPlayer);
    } catch (err) {
      setAudioStatus(`Error: ${err.message}`);
    }
  };

  const uploadAudioFile = async () => {
    const file = els.audioFile.files && els.audioFile.files[0];
    if (!file) { setAudioStatus("Choose an audio file first."); return; }
    setAudioStatus("Transcribing…");
    els.uploadAudioBtn.disabled = true;
    await sendAudioBlob(file, file.name);
    els.uploadAudioBtn.disabled = false;
  };

  const recordMicrophone = async () => {
    setAudioStatus(`Recording for ${state.recordingSeconds} seconds…`);
    els.recordAudioBtn.disabled = true;
    try {
      if (!state.audioStream) {
        state.audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      }
      const chunks  = [];
      const options = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? { mimeType: "audio/webm;codecs=opus" } : {};
      const recorder = new MediaRecorder(state.audioStream, options);
      recorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunks.push(e.data); };
      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
        await sendAudioBlob(blob, "recording.webm");
        els.recordAudioBtn.disabled = false;
      };
      recorder.start();
      setTimeout(() => {
        if (recorder.state !== "inactive") recorder.stop();
      }, state.recordingSeconds * 1000);
    } catch (err) {
      setAudioStatus(`Microphone error: ${err.message}`);
      els.recordAudioBtn.disabled = false;
    }
  };

  /* ─── TAB / MODE SWITCHING ─── */
  const switchMode = async (mode) => {
    if (state.activeMode === mode) return;
    state.activeMode = mode;

    /* update tab buttons */
    [els.tabSign, els.tabText, els.tabAudio].forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.mode === mode);
    });

    /* show/hide panels */
    els.modeSign.classList.toggle("hidden",  mode !== "sign");
    els.modeText.classList.toggle("hidden",  mode !== "text");
    els.modeAudio.classList.toggle("hidden", mode !== "audio");

    if (mode === "sign") {
      await startCamera();
    } else {
      stopCamera();
    }
  };

  /* ─── WIRE EVENTS ─── */
  [els.tabSign, els.tabText, els.tabAudio].forEach((btn) => {
    btn.addEventListener("click", () => switchMode(btn.dataset.mode));
  });

  els.clearHistoryBtn.addEventListener("click", clearHistory);
  els.speakTextBtn.addEventListener("click", speakText);
  els.showSignBtn.addEventListener("click", showSign);
  els.uploadAudioBtn.addEventListener("click", uploadAudioFile);
  els.recordAudioBtn.addEventListener("click", recordMicrophone);
  els.textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) speakText();
  });

  /* keep canvas synced if window resizes */
  window.addEventListener("resize", syncCanvasSize);

  /* ─── INIT ─── */
  renderModelInfo(appConfig.gesture_stats || {});
  loadHistory().catch(() => {});
  updateHistory([]);
  /* Start on sign mode */
  state.activeMode = ""; // force switchMode to actually run
  switchMode("sign");
})();
