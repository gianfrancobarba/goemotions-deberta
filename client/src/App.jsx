import { useState, useEffect } from "react";
import "./App.css";
import EmotionPieChart from "./components/EmotionPieChart";
import { FiSend } from "react-icons/fi";

export default function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState(() => {
    const stored = localStorage.getItem("chat_history");
    return stored
      ? JSON.parse(stored)
      : [
          {
            id: crypto.randomUUID(),
            type: "system",
            text: "Benvenuto! Inizia a scrivere una frase per analizzarne le emozioni.",
            timestamp: new Date(),
          },
        ];
  });

  // Salva in localStorage ogni volta che cambia messages
  useEffect(() => {
    localStorage.setItem("chat_history", JSON.stringify(messages));
  }, [messages]);

  const handleSubmit = async () => {
    if (!input.trim()) return;

    const userText = input;
    const newUserMessage = {
      id: crypto.randomUUID(),
      type: "user",
      text: userText,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setInput("");

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userText }),
      });

      const data = await response.json();
      const newBotMessage = {
        id: crypto.randomUUID(),
        type: "bot",
        sentiment: data.sentiment,
        emotions: data.emotions,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, newBotMessage]);
    } catch (err) {
      const errorMessage = {
        id: crypto.randomUUID(),
        type: "bot",
        text: "Errore nella chiamata API.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
      console.error("Errore API:", err);
    }
  };

  const userHistory = messages
    .filter((msg) => msg.type === "user")
    .slice()
    .reverse(); // Mostra le ultime in alto

  return (
    <div className="layout">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <button
          className="button"
          onClick={() => {
            setMessages([]);
            localStorage.removeItem("chat_history");
          }}
        >
          Nuova analisi
        </button>
        <div className="history-title">CRONOLOGIA ANALISI</div>
        <div className="history-scroll">
          {userHistory.map((msg) => (
            <div className="history-item" key={msg.id}>
              <span>
                {msg.text.slice(0, 40)}
                {msg.text.length > 40 && "..."}
              </span>
              <span className="history-date">
                {new Date(msg.timestamp).toLocaleDateString()}
              </span>
            </div>
          ))}
        </div>
      </aside>

      {/* MAIN */}
      <div className="main-section">
        <header className="header">
          <h1 className="main-title">Analisi delle Emozioni</h1>
          <p className="intro-description">
            L'AI analizzerà il testo ed estrarrà le emozioni contenute in un
            grafico interattivo. I risultati sono generati da un modello AI e
            potrebbero non essere sempre accurati.
          </p>
        </header>

        <main className="chat-content">
          {messages.map((msg) => (
            <div key={msg.id} className={`message-bubble ${msg.type}`}>
              {msg.type === "bot" && msg.emotions ? (
                <>
                  <p>
                    <strong>Sentimento:</strong> {msg.sentiment.toUpperCase()}
                  </p>
                  <EmotionPieChart emotions={msg.emotions} />
                </>
              ) : (
                <p>{msg.text}</p>
              )}
            </div>
          ))}
        </main>

        <footer className="chat-input-wrapper">
          <div className="chat-input">
            <textarea
              ref={(el) => {
                if (el) {
                  el.style.height = "auto";
                  el.style.height = Math.min(el.scrollHeight, 180) + "px";
                }
              }}
              rows="1"
              placeholder="Inserisci il tuo testo qui..."
              className="input-textarea"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onInput={(e) => {
                e.target.style.height = "auto";
                e.target.style.height =
                  Math.min(e.target.scrollHeight, 180) + "px";
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
            />

            <button className="send-button" onClick={handleSubmit}>
              <FiSend size={18} />
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}
