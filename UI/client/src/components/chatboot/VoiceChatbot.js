import React, { useState, useEffect, useRef } from 'react';
import stringSimilarity from 'string-similarity';

// Predefined FAQs and answers
const predefinedAnswers = {
  "how do i play the space invader game using my body": "To play Space Invader, raise your arms to move left or right and step forward to shoot.",
  "what poses control the flappy bird character": "For Flappy Bird, jumping or raising your hands makes the bird flap.",
  "how do i slice fruit in the ninja fruit game": "Move your hand quickly across the screen to slice the fruit!",
  "does the camera need to see my full body": "It works best if the camera sees your upper body and hands.",
  "what should i do if my movements arent recognized": "Make sure there's enough light and your camera can clearly see your body.",
  "how does the app understand my gestures": "It uses AI to analyze your body movements and match them to game controls.",
  "can i play all games using just hand movements": "Yes! All games are designed to work with simple hand gestures.",
  "what pose launches a missile in the space invader game": "Extend both arms forward to launch a missile.",
  "do i need a lot of space to play": "A small area where your upper body is visible to the camera is enough.",
  "can i use voice to switch between games or levels": "Currently, voice is used for asking questions, but future updates may include voice-based game switching!",
};

const VoiceChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError] = useState(null);
  const recognitionRef = useRef(null);
  const chatContainerRef = useRef(null);
  const lastRequestTime = useRef(0);
  const debounceTimeout = useRef(null);

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setError('Speech recognition is not supported in this browser.');
      return;
    }

    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous = false;
    recognitionRef.current.interimResults = false;
    recognitionRef.current.lang = 'en-US';

    recognitionRef.current.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      if (event.results[0].isFinal) {
        if (debounceTimeout.current) clearTimeout(debounceTimeout.current);
        debounceTimeout.current = setTimeout(() => {
          handleUserMessage(transcript);
        }, 1000);
      }
    };

    recognitionRef.current.onerror = (event) => {
      setError(`Speech recognition error: ${event.error}`);
      setIsListening(false);
    };

    recognitionRef.current.onend = () => {
      setIsListening(false);
    };

    return () => {
      recognitionRef.current?.stop();
    };
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // --- Changed: Use FastAPI backend for LLM and TTS ---
  const handleUserMessage = async (text) => {
    if (!text.trim()) return;

    const userMessage = { id: `user-${Date.now()}`, text, isUser: true };
    setMessages((prev) => [...prev, userMessage]);

    // Normalize text for comparison
    const normalizedText = text.toLowerCase().replace(/[^\w\s]/g, '');
    const keys = Object.keys(predefinedAnswers);
    const { bestMatch } = stringSimilarity.findBestMatch(normalizedText, keys);

    if (bestMatch.rating > 0.75) {
      const predefinedResponse = predefinedAnswers[bestMatch.target];
      const aiMessage = { id: `ai-${Date.now()}`, text: predefinedResponse, isUser: false };
      setMessages((prev) => [...prev, aiMessage]);
      await playTTS(predefinedResponse); // Use backend TTS
      return;
    }

    // Otherwise call FastAPI LLM backend
    const now = Date.now();
    const timeSinceLastRequest = now - lastRequestTime.current;
    if (timeSinceLastRequest < 1000) await sleep(1000 - timeSinceLastRequest);
    lastRequestTime.current = now;

    let llmData;
    try {
      const res = await fetch("http://localhost:8000/api/llm/reply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text }),
      });
      if (!res.ok) throw new Error("LLM API error");
      llmData = await res.json(); // { response, cleaned_for_tts }
    } catch (err) {
      setMessages((prev) => [...prev, { id: `err-${Date.now()}`, text: "LLM backend error.", isUser: false }]);
      return;
    }

    const aiMessage = { id: `ai-${Date.now()}`, text: llmData.response, isUser: false };
    setMessages((prev) => [...prev, aiMessage]);

    await playTTS(llmData.cleaned_for_tts || llmData.response);
  };

  // --- Changed: Use backend TTS API ---
  async function playTTS(text, voice = "af_heart", speed = 1.15, lang = "en-us") {
    try {
      const response = await fetch("http://localhost:8000/api/tts/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice, speed, lang }),
      });
      if (!response.ok) throw new Error("TTS API error");
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      setIsSpeaking(true);
      audio.onended = () => setIsSpeaking(false);
      audio.play();
    } catch (err) {
      setMessages((prev) => [...prev, { id: `err-${Date.now()}`, text: "TTS backend error.", isUser: false }]);
      setIsSpeaking(false);
    }
  }

  const toggleListening = () => {
    if (!recognitionRef.current) return;
    if (isListening) {
      recognitionRef.current.stop();
    } else {
      try {
        recognitionRef.current.start();
        setIsListening(true);
        setError(null);
      } catch (error) {
        setError('Microphone access denied. Please allow microphone access.');
        setIsListening(false);
      }
    }
  };

  return (
    <div style={styles.container}>
      {/* LEFT FAQ PANEL */}
      <div style={styles.faqPanel}>
        <h2 style={styles.panelTitle}>📘 FAQs</h2>
        <ul style={styles.faqList}>
          <li>🤖 How do I play the Space Invader game using my body?</li>
          <li>🕹️ What poses control the Flappy Bird character?</li>
          <li>🍉 How do I slice fruit in the Ninja Fruit game?</li>
          <li>📷 Does the camera need to see my full body?</li>
          <li>⚠️ What should I do if my movements aren't recognized?</li>
          <li>🧠 How does the app understand my gestures?</li>
          <li>🎯 Can I play all games using just hand movements?</li>
          <li>🚀 What pose launches a missile in the Space Invader game?</li>
          <li>🕴️ Do I need a lot of space to play?</li>
          <li>🗣️ Can I use voice to switch between games or levels?</li>
        </ul>
      </div>

      {/* RIGHT CHAT PANEL */}
      <div style={styles.chatPanel}>
        <div style={styles.header}>🎤 Voice Chat with ARISE Bot</div>

        <div style={styles.messages} ref={chatContainerRef}>
          {messages.length === 0 && (
            <div style={styles.welcome}>
              Welcome! Click the microphone to begin speaking.
            </div>
          )}
          {messages.map((msg) => (
            <div
              key={msg.id}
              style={{
                ...styles.message,
                alignSelf: msg.isUser ? 'flex-end' : 'flex-start',
                backgroundColor: msg.isUser ? '#dcf8c6' : '#f1f0f0',
              }}
            >
              {msg.text}
            </div>
          ))}
        </div>

        {error && <div style={styles.error}>{error}</div>}

        <div style={styles.controls}>
          <button
            className={isListening ? 'mic-button-listening' : 'mic-button'}
            onClick={toggleListening}
            style={{
              ...styles.micButton,
            }}
            disabled={isSpeaking}
          >
            {isListening ? 'Stop' : 'Speak'}
          </button>
          {isSpeaking && <span style={styles.status}>BlueJay Boot is speaking...</span>}
        </div>
      </div>
    </div>
  );
};

export default VoiceChatbot;

// Styles unchanged
const styles = {
  container: {
    display: 'flex',
    height: '70vh',
    fontFamily: 'Arial, sans-serif',
    backgroundColor: '#f5f5f5',
    marginTop: '10px',
  },
  faqPanel: {
    width: '30%',
    backgroundColor: '#e3eaf2',
    padding: '2rem',
    boxShadow: '2px 0 5px rgba(0,0,0,0.1)',
  },
  panelTitle: {
    marginBottom: '1rem',
    fontSize: '1.5rem',
  },
  faqList: {
    listStyle: 'disc inside',
    paddingLeft: '1rem',
    fontSize: '1rem',
    color: '#333',
  },
  chatPanel: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    padding: '1rem',
    backgroundColor: '#ffffff',
  },
  header: {
    fontSize: '1.3rem',
    fontWeight: 'bold',
    marginBottom: '1rem',
  },
  messages: {
    flex: 1,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.6rem',
    padding: '1rem',
    border: '1px solid #ddd',
    borderRadius: '8px',
    backgroundColor: '#fafafa',
  },
  message: {
    padding: '10px 14px',
    borderRadius: '18px',
    maxWidth: '70%',
    fontSize: '1rem',
    lineHeight: '1.4',
  },
  welcome: {
    textAlign: 'center',
    color: '#888',
    fontSize: '1rem',
  },
  controls: {
    display: 'flex',
    alignItems: 'center',
    marginTop: '1rem',
    gap: '1rem',
  },
  micButton: {
    padding: '10px 18px',
    fontSize: '1rem',
    color: '#fff',
    border: 'none',
    borderRadius: '30px',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
  },
  status: {
    fontSize: '0.9rem',
    color: '#666',
  },
  error: {
    marginTop: '1rem',
    color: 'red',
    fontSize: '0.9rem',
  },
};