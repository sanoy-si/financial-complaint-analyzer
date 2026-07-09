"use client";

import { useEffect, useReducer, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";

const SCRIPT = [
  {
    q: "What are the most common complaint types in Q3?",
    a: "Based on the indexed reports, the top complaint categories in Q3 were:\n1. Billing errors (34%)\n2. Unauthorized charges (28%)\n3. Account access issues (19%)\n\nCredit card complaints rose 12% vs Q2.",
    cite: "CFPB Q3 Report · p.14",
  },
  {
    q: "Which financial products have the highest resolution rate?",
    a: "Mortgage complaints show the highest resolution rate at 78%, followed by auto loans at 71%. Credit card disputes tend to resolve slowest, averaging 42 days to close.",
    cite: "Complaint Database 2024",
  },
  {
    q: "Summarise the GDPR implications for data retention",
    a: "Under GDPR Article 5(1)(e), personal data must not be kept longer than necessary. For financial records, typical retention is 5–7 years, but complaint data may require shorter windows unless actively referenced.",
    cite: "Policy Handbook · §3.2",
  },
];

type State = {
  scriptIdx: number;
  phase: "typing-q" | "pausing" | "typing-a" | "done" | "waiting";
  q: string;
  a: string;
  showCite: boolean;
};

type Action =
  | { type: "tick-q"; char: string }
  | { type: "start-a" }
  | { type: "tick-a"; char: string }
  | { type: "show-cite" }
  | { type: "next" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "tick-q":  return { ...state, q: state.q + action.char };
    case "start-a": return { ...state, phase: "typing-a" };
    case "tick-a":  return { ...state, a: state.a + action.char };
    case "show-cite": return { ...state, showCite: true, phase: "done" };
    case "next": {
      const next = (state.scriptIdx + 1) % SCRIPT.length;
      return { scriptIdx: next, phase: "typing-q", q: "", a: "", showCite: false };
    }
    default: return state;
  }
}

export function TypingDemo() {
  const [state, dispatch] = useReducer(reducer, {
    scriptIdx: 0, phase: "typing-q", q: "", a: "", showCite: false,
  });
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const { scriptIdx, phase, q, a } = state;
    const item = SCRIPT[scriptIdx];

    function clear() { if (timerRef.current) clearTimeout(timerRef.current); }

    if (phase === "typing-q") {
      if (q.length < item.q.length) {
        timerRef.current = setTimeout(
          () => dispatch({ type: "tick-q", char: item.q[q.length] }),
          32 + Math.random() * 28,
        );
      } else {
        timerRef.current = setTimeout(() => dispatch({ type: "start-a" }), 500);
      }
    } else if (phase === "typing-a") {
      if (a.length < item.a.length) {
        timerRef.current = setTimeout(
          () => dispatch({ type: "tick-a", char: item.a[a.length] }),
          14 + Math.random() * 12,
        );
      } else {
        timerRef.current = setTimeout(() => dispatch({ type: "show-cite" }), 300);
      }
    } else if (phase === "done") {
      timerRef.current = setTimeout(() => dispatch({ type: "next" }), 4200);
    }

    return clear;
  }, [state]);

  const item = SCRIPT[state.scriptIdx];

  return (
    <div className="demo-shell">
      <div className="demo-titlebar">
        <span className="demo-dot demo-dot-r" />
        <span className="demo-dot demo-dot-y" />
        <span className="demo-dot demo-dot-g" />
        <span className="demo-title">Grounded Chat · {item.cite.split("·")[0].trim()}</span>
      </div>
      <div className="demo-body">
        <AnimatePresence mode="wait">
          <motion.div
            key={state.scriptIdx + "-q"}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="demo-msg user"
          >
            {state.q}
            {state.phase === "typing-q" && <span className="cursor" />}
          </motion.div>
        </AnimatePresence>

        {state.phase !== "typing-q" && (
          <AnimatePresence>
            <motion.div
              key={state.scriptIdx + "-a"}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="demo-msg bot"
            >
              {state.a}
              {state.phase === "typing-a" && <span className="cursor" />}
              {state.showCite && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.15 }}
                >
                  <span className="demo-cite">📎 {item.cite}</span>
                </motion.div>
              )}
            </motion.div>
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}
