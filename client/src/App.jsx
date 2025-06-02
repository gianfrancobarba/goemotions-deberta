// File: App.jsx

import { useState, useEffect } from "react";
import "./App.css";
import {
  ResponsiveContainer,
  BarChart,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Bar,
  Cell,
} from "recharts";
import { FiSend } from "react-icons/fi";

//
// 1) DEFINIZIONE DEI GRUPPI EMOZIONALI
//
const POSITIVE_SET = new Set([
  "joy",
  "love",
  "gratitude",
  "relief",
  "amusement",
  "optimism",
  "admiration",
  "pride",
  "approval",
  "caring",
  "excitement",
  "contentment",
]);

const NEGATIVE_SET = new Set([
  "anger",
  "fear",
  "sadness",
  "disappointment",
  "disgust",
  "remorse",
  "grief",
  "annoyance",
  "disapproval",
  "embarrassment",
  "nervousness",
]);

const NEUTRAL_SET = new Set([
  "realization",
  "curiosity",
  "confusion",
  "surprise",
]);

// Colori per ciascun gruppo
const COLOR_POS = "#4ade80"; // verde chiaro
const COLOR_NEG = "#f87171"; // rosso chiaro
const COLOR_NEU = "#60a5fa"; // blu chiaro

function getColorByEmotion(name) {
  if (POSITIVE_SET.has(name)) return COLOR_POS;
  if (NEGATIVE_SET.has(name)) return COLOR_NEG;
  if (NEUTRAL_SET.has(name)) return COLOR_NEU;
  return "#6b7280"; // grigio di default
}

//
// 2) COMPONENTE: TypingIndicator
//
function TypingIndicator() {
  return (
      <div className="typing-indicator">
        <span className="dot dot1" />
        <span className="dot dot2" />
        <span className="dot dot3" />
      </div>
  );
}

//
// 3) COMPONENTE: Tooltip personalizzato per Recharts
//
function CustomTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    const { name, value } = payload[0].payload;
    return (
        <div
            style={{
              background: "#1a1f2b",
              border: "1px solid #2c3342",
              padding: "8px 12px",
              borderRadius: "6px",
              color: "#f1f5f9",
              fontSize: "0.9rem",
              boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
            }}
        >
          <div style={{ marginBottom: "4px", fontWeight: "600" }}>
            {name.charAt(0).toUpperCase() + name.slice(1)}
          </div>
          <div>
            Valore: <strong>{value.toFixed(2)}</strong>
          </div>
        </div>
    );
  }
  return null;
}

//
// 4) COMPONENTE: EmotionBarChart
//
function EmotionBarChart({ emotions }) {
  // Converto { name: value, … } → [{ name, value }, … ] e ordino decrescente
  const rawData = Object.entries(emotions).map(([name, value]) => ({ name, value }));
  const sortedData = rawData.sort((a, b) => b.value - a.value);

  return (
      <div style={{ width: "100%", height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
              layout="vertical"
              data={sortedData}
              margin={{ top: 20, right: 20, left: 40, bottom: 20 }}
          >
            <CartesianGrid vertical={false} stroke="#2c3342" strokeOpacity={0.3} />
            <YAxis
                dataKey="name"
                type="category"
                tick={{ fill: "#cbd5e1", fontSize: 12 }}
                width={110}
            />
            <XAxis
                type="number"
                domain={[0, 1]}
                tick={{ fill: "#cbd5e1", fontSize: 12 }}
                axisLine={{ stroke: "#2c3342" }}
                tickLine={false}
            />
            <Tooltip content={CustomTooltip} cursor={{ fill: "rgba(255,255,255,0.05)" }} />
            <Bar dataKey="value" barSize={24} animationDuration={1500} isAnimationActive={true}>
              {sortedData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getColorByEmotion(entry.name)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
  );
}

//
// 5) COMPONENTE PRINCIPALE: App
//
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

  // Stato per il typing indicator
  const [isLoading, setIsLoading] = useState(false);

  // Stato per il toggle dei dettagli: mappa "msgId-emotion" → booleano
  const [showDetails, setShowDetails] = useState({});

  // Salvo in localStorage ogni volta che cambia 'messages'
  useEffect(() => {
    localStorage.setItem("chat_history", JSON.stringify(messages));
  }, [messages]);

  const handleSubmit = async () => {
    if (!input.trim()) return;

    const userText = input;
    // Aggiungo subito il messaggio dell'utente
    const newUserMessage = {
      id: crypto.randomUUID(),
      type: "user",
      text: userText,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, newUserMessage]);
    setInput("");

    setIsLoading(true);
    try {
      // 1) Chiamata a /predict
      const predictRes = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userText }),
      });
      const predictData = await predictRes.json();

      // 2) Chiamata a /explain
      const explainRes = await fetch("http://localhost:8000/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userText }),
      });
      const explainData = await explainRes.json();

      // Costruisco il messaggio bot con tutti i campi
      const newBotMessage = {
        id: crypto.randomUUID(),
        type: "bot",
        sentiment: predictData.sentiment,
        emotions: predictData.emotions,
        explanations: explainData.explanations,
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
    } finally {
      setIsLoading(false);
    }
  };

  const userHistory = messages
      .filter((msg) => msg.type === "user")
      .slice()
      .reverse();

  return (
      <div className="layout">
        {/* === SIDEBAR === */}
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

        {/* === MAIN SECTION === */}
        <div className="main-section">
          <header className="header">
            <h1 className="main-title">Analisi delle Emozioni</h1>
            <p className="intro-description">
              L'AI analizzerà il testo ed estrarrà le emozioni contenute in un
              grafico interattivo e fornirà una spiegazione dettagliata del
              modello. I risultati sono generati da un modello AI e potrebbero
              non essere sempre accurati.
            </p>
          </header>

          <main className="chat-content">
            {messages.map((msg) => (
                <div key={msg.id} className={`message-bubble ${msg.type}`}>
                  {msg.type === "bot" && msg.emotions ? (
                      <>
                        {/* === BLOCCO SPIEGAZIONI STRUTTURATE === */}
                        {msg.explanations && (
                            <div className="explanation-section">
                              <h4>Spiegazione del modello:</h4>

                              {Object.entries(msg.explanations).map(([emo, detail]) => {
                                // Chiave per gestire lo stato di toggle di questa emozione
                                const toggleKey = `${msg.id}-${emo}`;

                                return (
                                    <div key={toggleKey} className="explanation-block">
                                      {/* ===== 1) Simple explanation (multilinea) ===== */}
                                      <pre className="simple-explanation">
{detail.simple_explanation}
                            </pre>
                                      {/* Notare l’uso di <pre> per preservare i newline */}

                                      {/* ===== 2) Bottone per mostrare/nascondere dettagli ===== */}
                                      <button
                                          className="toggle-details"
                                          aria-expanded={showDetails[toggleKey] ? "true" : "false"}
                                          onClick={() =>
                                              setShowDetails((prev) => ({
                                                ...prev,
                                                [toggleKey]: !prev[toggleKey],
                                              }))
                                          }
                                      >
                                        {showDetails[toggleKey]
                                            ? "Nascondi dettagli avanzati"
                                            : "Mostra dettagli avanzati"}
                                      </button>

                                      {/* ===== 3) Sezione dettagli avanzati ===== */}
                                      {showDetails[toggleKey] && (
                                          <div className="detailed-explanation">
                                            {/* 3.1) Probabilità */}
                                            <div className="detail-item">
                                              <strong>Probabilità:</strong>{" "}
                                              {detail.probability.toFixed(2)}
                                            </div>

                                            {/* 3.2) Feature Importances */}
                                            <div className="detail-item">
                                              <strong>Feature più rilevanti:</strong>
                                              <table className="imp-table">
                                                <thead>
                                                <tr>
                                                  <th>Feature</th>
                                                  <th>Peso</th>
                                                </tr>
                                                </thead>
                                                <tbody>
                                                {detail.all_importances.map(
                                                    ([feat, imp], idx) => (
                                                        <tr key={idx}>
                                                          <td>{feat}</td>
                                                          <td>{imp.toFixed(4)}</td>
                                                        </tr>
                                                    )
                                                )}
                                                </tbody>
                                              </table>
                                              <small className="note">
                                                Il “peso” indica quanto ciascuna caratteristica ha
                                                influito sulla decisione.
                                              </small>
                                            </div>

                                            {/* 3.3) Dettagli del testo */}
                                            <div className="detail-item">
                                              <strong>Dettagli del testo:</strong>
                                              <ul>
                                                <li>
                                                  Numero parole totali:{" "}
                                                  <strong>{detail.word_count}</strong>
                                                </li>

                                                {detail.emotion_words &&
                                                detail.emotion_words.length > 0 ? (
                                                    <li>
                                                      Parole emozionali:{" "}
                                                      <strong>
                                                        {detail.emotion_words.join(", ")}
                                                      </strong>
                                                    </li>
                                                ) : (
                                                    <li>Nessuna parola emotiva rilevata</li>
                                                )}

                                                {detail.stopword_ratio !== null && (
                                                    <li>
                                                      Rapporto stopword:{" "}
                                                      <strong>
                                                        {detail.stopword_ratio.toFixed(2)}
                                                      </strong>{" "}
                                                      <small>
                                                        (rapporto tra parole neutre e parole
                                                        emotive)
                                                      </small>
                                                    </li>
                                                )}

                                                {detail.has_negation && (
                                                    <li>Presenza di negazioni.</li>
                                                )}

                                                {detail.exclamation_count > 0 && (
                                                    <li>
                                                      {detail.exclamation_count === 1
                                                          ? "1 punto esclamativo"
                                                          : `${detail.exclamation_count} punti esclamativi`}
                                                    </li>
                                                )}

                                                {detail.question_count > 0 && (
                                                    <li>
                                                      {detail.question_count === 1
                                                          ? "1 punto interrogativo"
                                                          : `${detail.question_count} punti interrogativi`}
                                                    </li>
                                                )}
                                              </ul>
                                            </div>

                                            {/* 3.4) Metriche surrogate */}
                                            <div className="detail-item">
                                              <strong>
                                                Metriche surrogate (decision tree semplificato):
                                              </strong>
                                              <ul>
                                                {detail.metrics.fidelity !== null && (
                                                    <li>
                                                      Fidelity:{" "}
                                                      <strong>
                                                        {detail.metrics.fidelity.toFixed(2)}
                                                      </strong>{" "}
                                                      <small>
                                                        (quanto l’albero spiega il modello
                                                        principale)
                                                      </small>
                                                    </li>
                                                )}
                                                {detail.metrics.sparsity !== null && (
                                                    <li>
                                                      Sparsità:{" "}
                                                      <strong>
                                                        {detail.metrics.sparsity.toFixed(2)}
                                                      </strong>{" "}
                                                      <small>
                                                        (numero di feature usate dall'albero)
                                                      </small>
                                                    </li>
                                                )}
                                                {detail.metrics.stability !== null && (
                                                    <li>
                                                      Stabilità:{" "}
                                                      <strong>
                                                        {detail.metrics.stability.toFixed(2)}
                                                      </strong>{" "}
                                                      <small>
                                                        (robustezza delle regole alle
                                                        variazioni dei dati)
                                                      </small>
                                                    </li>
                                                )}
                                              </ul>
                                            </div>

                                            {/* 3.5) Regole surrogate (albero decisionale) */}
                                            {detail.parsed_rules && (
                                                <div className="detail-item rules-block">
                                                  <strong>
                                                    Regole surrogate (decision tree):
                                                  </strong>
                                                  <pre className="rules-pre">
{detail.parsed_rules}
                                    </pre>
                                                  <small className="note">
                                                    Struttura dell’albero surrogate.
                                                  </small>
                                                </div>
                                            )}
                                          </div>
                                      )}
                                    </div>
                                );
                              })}
                            </div>
                        )}

                        {/* === SENTIMENTO E GRAFICO === */}
                        <p>
                          <strong>Sentimento:</strong> {msg.sentiment.toUpperCase()}
                        </p>
                        <EmotionBarChart emotions={msg.emotions} />
                      </>
                  ) : (
                      <p>{msg.text}</p>
                  )}
                </div>
            ))}

            {/* Indicatore di “bot is typing” */}
            {isLoading && (
                <div className="message-bubble bot">
                  <TypingIndicator />
                </div>
            )}
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
                  e.target.style.height = Math.min(e.target.scrollHeight, 180) + "px";
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